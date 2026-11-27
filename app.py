import datetime
import logging
import re
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =========================================================
# 頁面基本設定與黑底質感 CSS（手機優先排版）
# =========================================================
st.set_page_config(
    page_title="台股量化風控儀表板",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    .risk-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .card-title { font-size: 0.85rem; color: #8b949e; margin-bottom: 4px; }
    .card-value { font-size: 1.85rem; font-weight: 700; line-height: 1.2; }
    .card-sub { font-size: 0.82rem; color: #8b949e; margin-top: 6px; line-height: 1.5; }
    .badge-red { background: rgba(255,123,114,0.18); color: #ff7b72; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-yellow { background: rgba(210,153,34,0.20); color: #e3b341; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-green { background: rgba(63,185,80,0.18); color: #3fb950; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .m-label { font-size: 0.72rem; color: #8b949e; }
    .m-val { font-size: 1.05rem; font-weight: 700; color: #f0f6fc; margin: 2px 0; }
    .m-tag { font-size: 0.7rem; font-weight: 600; }
    .tag-good { color: #3fb950; }
    .tag-mid { color: #e3b341; }
    .tag-bad { color: #ff7b72; }
    .health-list { font-size: 0.84rem; color: #c9d1d9; margin-top: 6px; line-height: 1.6; }
    .disclaimer { font-size: 0.75rem; color: #6e7681; text-align: center; margin-top: 24px; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 內建熱門台股中文名稱備援表（當 API 限流時仍可正確顯示中文）
FALLBACK_STOCK_NAMES = {
    "2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2382": "廣達", "2308": "台達電",
    "2303": "聯電", "3711": "日月光投控", "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
    "2886": "兆豐金", "2884": "玉山金", "2383": "台光電", "6274": "台燿", "8046": "南電",
    "1815": "富喬", "3231": "緯創", "6669": "緯穎", "2357": "華碩", "2376": "技嘉",
    "3017": "奇鋐", "3324": "雙鴻", "2603": "長榮", "2609": "陽明", "2615": "萬海",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益台灣精選高息",
}


# =========================================================
# 工具函式與資料結構
# =========================================================
def get_finmind_token() -> str:
    """優先讀取 Streamlit Cloud Secrets，其次讀取側邊欄手動輸入"""
    if st.session_state.get("user_finmind_token"):
        return st.session_state["user_finmind_token"].strip()
    try:
        return st.secrets.get("FINMIND_TOKEN", "").strip()
    except Exception:
        return ""


def round_tw_tick(price: float) -> float:
    if pd.isna(price) or price <= 0:
        return 0.0
    if price < 10:
        tick = 0.01
    elif price < 50:
        tick = 0.05
    elif price < 100:
        tick = 0.1
    elif price < 500:
        tick = 0.5
    elif price < 1000:
        tick = 1.0
    else:
        tick = 5.0
    return round(round(price / tick) * tick, 2)


def _fmt_num(val: Optional[float], suffix: str = "", signed: bool = False) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:+.1f}{suffix}" if signed else f"{val:.1f}{suffix}"


def get_traffic_light(score: int) -> Tuple[str, str, str, str]:
    """回傳 (icon, 狀態文字, 顏色HEX, CSS徽章類別)"""
    if score >= 65:
        return "🔴", "危險", "#ff7b72", "badge-red"
    elif score >= 48:
        return "🟡", "中等", "#e3b341", "badge-yellow"
    return "🟢", "安全", "#3fb950", "badge-green"


@dataclass
class GoldenRatioWeights:
    # 第一層：大盤系統風險 (MRI) 權重
    w_us_tech: float = 0.200
    w_rates_fx: float = 0.180
    w_tw_market: float = 0.238
    w_commodities: float = 0.120
    w_btc_liquidity: float = 0.100
    w_vix_news: float = 0.162
    # 第二層：個股風險 (SRI) 權重 — 大盤 40% ＋ 個股（籌碼與體質）60%
    w_mri_transmission: float = 0.400
    w_idiosyncratic: float = 0.600
    w_stock_chip_in_idio: float = 0.618
    w_stock_health_in_idio: float = 0.382


@dataclass
class RiskModelConfig:
    window_days: int = 10
    chip_smooth_days: int = 5
    weights: GoldenRatioWeights = field(default_factory=GoldenRatioWeights)
    lock_floor_level1: float = 68.0
    lock_floor_level2: float = 78.0


@dataclass
class FundamentalQualityReport:
    quality_score: int
    risk_equivalent: float
    rating: str
    pe: Optional[float] = None
    pb: Optional[float] = None
    roe_pct: Optional[float] = None
    op_margin_pct: Optional[float] = None
    rev_growth_pct: Optional[float] = None
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =========================================================
# 後台消息面與資料擷取層（附 Streamlit 雲端快取）
# =========================================================
BEARISH_KW = {
    "升息": 3.0, "通膨": 2.5, "關稅": 3.0, "制裁": 2.5, "禁令": 2.5,
    "重挫": 3.0, "崩跌": 3.5, "暴跌": 3.5, "斷頭": 3.5, "衰退": 3.0,
    "停滯": 2.5, "戰爭": 3.0, "衝突": 2.5, "賣超": 2.0, "提款": 2.0,
    "緊縮": 2.5, "鷹派": 2.5, "下修": 2.0, "恐慌": 2.5, "破底": 2.5,
    "大跌": 2.5, "利空": 2.0, "匯出": 2.0,
}
BULLISH_KW = {
    "降息": 3.0, "鴿派": 2.5, "寬鬆": 2.5, "創高": 2.0, "買超": 2.0,
    "回補": 2.0, "強彈": 2.0, "大漲": 2.0, "利多": 2.0, "資本支出": 2.0,
    "擴產": 2.0, "和談": 2.5, "停火": 3.0, "上修": 2.0, "優於預期": 2.5,
    "護盤": 2.0, "景氣回溫": 2.0,
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tw_stock_name_map(token: str) -> Dict[str, str]:
    """每日快取一次全台股上市櫃中文名稱對照表"""
    name_map = dict(FALLBACK_STOCK_NAMES)
    params = {"dataset": "TaiwanStockInfo"}
    if token:
        params["token"] = token
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("status") == 200 and j.get("data"):
                for item in j["data"]:
                    sid = str(item.get("stock_id", "")).strip()
                    sname = str(item.get("stock_name", "")).strip()
                    if sid and sname:
                        name_map[sid] = sname
    except Exception:
        pass
    return name_map


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_risk_score(stock_id: str = "") -> float:
    query = "台股 OR 美股 OR 聯準會 OR 美債 OR 關稅"
    if stock_id:
        query += f" OR {stock_id}"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return 50.0
        items = ET.fromstring(resp.content).findall(".//item")[:25]
        if not items:
            return 50.0
        bear = sum(w for it in items for kw, w in BEARISH_KW.items() if kw in (it.findtext("title") or ""))
        bull = sum(w for it in items for kw, w in BULLISH_KW.items() if kw in (it.findtext("title") or ""))
        if bear + bull == 0:
            return 50.0
        net_ratio = (bear - bull) / max(bear + bull, 8.0)
        return float(np.clip(50.0 + net_ratio * 35.0, 20.0, 85.0))
    except Exception:
        return 50.0


def _fetch_yf(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def _finmind_get(dataset: str, start: str, end: str, token: str, data_id: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    params = {"dataset": dataset, "start_date": start, "end_date": end}
    if data_id:
        params["data_id"] = data_id
    if token:
        params["token"] = token
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("status") == 200 and j.get("data"):
                return pd.DataFrame(j["data"]), "OK"
            return pd.DataFrame(), str(j.get("msg", "無資料"))
        return pd.DataFrame(), f"HTTP {r.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_shared_market_data(fetch_start: str, end: str, fm_start: str, token: str):
    """快取全市場共用總經與大盤籌碼（每小時更新一次）"""
    twii = _fetch_yf("^TWII", fetch_start, end)
    macro_tickers = {
        "NDX": "^IXIC", "SOX": "^SOX", "TNX": "^TNX", "USDTWD": "TWD=X",
        "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD", "VIX": "^VIX",
    }
    macro_data = {}
    for k, sym in macro_tickers.items():
        d = _fetch_yf(sym, fetch_start, end)
        if k == "TNX" and not d.empty and d["Close"].median() > 20:
            d["Close"] = d["Close"] / 10.0
        macro_data[k] = d

    frames, mkt_status = [], "OK"
    df_tm, st1 = _finmind_get("TaiwanStockTotalMarginPurchaseShortSale", fm_start, end, token)
    if st1 != "OK":
        mkt_status = st1
    if not df_tm.empty and "name" in df_tm and "TodayBalance" in df_tm:
        df_tm["date"] = pd.to_datetime(df_tm["date"])
        sub = df_tm[df_tm["name"] == "MarginPurchaseMoney"]
        if sub.empty:
            sub = df_tm[df_tm["name"] == "MarginPurchase"]
        if not sub.empty:
            s = sub.set_index("date")["TodayBalance"].apply(pd.to_numeric, errors="coerce")
            frames.append(s.rename("mkt_margin_bal").to_frame())

    df_ti, st2 = _finmind_get("TaiwanStockTotalInstitutionalInvestors", fm_start, end, token)
    if st2 != "OK" and mkt_status == "OK":
        mkt_status = st2
    if not df_ti.empty and "name" in df_ti:
        df_ti["date"] = pd.to_datetime(df_ti["date"])
        df_ti["net_b"] = (
            pd.to_numeric(df_ti["buy"], errors="coerce").fillna(0)
            - pd.to_numeric(df_ti["sell"], errors="coerce").fillna(0)
        ) / 1e8
        piv = df_ti.pivot_table(index="date", columns="name", values="net_b", aggfunc="sum").fillna(0)
        if "Foreign_Investor" in piv:
            frames.append(piv[["Foreign_Investor"]].rename(columns={"Foreign_Investor": "mkt_foreign_net_b"}))

    mkt_chips = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return twii, macro_data, mkt_chips, mkt_status


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single_stock_bundle(stock_id: str, fetch_start: str, end: str, fm_start: str, token: str):
    clean_id = str(stock_id).strip().upper().replace(".TW", "").replace(".TWO", "")
    stk, resolved_sym = pd.DataFrame(), clean_id
    for suffix in [".TW", ".TWO"]:
        sym = f"{clean_id}{suffix}"
        df = _fetch_yf(sym, fetch_start, end)
        if not df.empty and len(df) > 10:
            stk, resolved_sym = df, sym
            break

    # 取得中文股名（優先 FinMind 全台股表 -> 次選 Yahoo shortName）
    name_map = fetch_tw_stock_name_map(token)
    stock_name = name_map.get(clean_id, "")

    # 財務體質健檢
    score = 52
    strengths, warns = [], []
    pe = pb = roe = op_m = rev_g = None
    try:
        info = yf.Ticker(resolved_sym).info or {}
        if not stock_name:
            stock_name = info.get("shortName") or info.get("longName") or ""
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        roe_r, op_r, rev_r = info.get("returnOnEquity"), info.get("operatingMargins"), info.get("revenueGrowth")
        roe = round(roe_r * 100, 1) if roe_r is not None else None
        op_m = round(op_r * 100, 1) if op_r is not None else None
        rev_g = round(rev_r * 100, 1) if rev_r is not None else None

        if roe is not None:
            if roe >= 15.0:
                score += 14
                strengths.append(f"股東報酬優異 (ROE {roe}%)")
            elif roe >= 8.0:
                score += 6
                strengths.append(f"獲利能力穩健 (ROE {roe}%)")
            elif roe < 0:
                score -= 16
                warns.append(f"目前處於虧損狀態 (ROE {roe}%)")
        if op_m is not None:
            if op_m >= 15.0:
                score += 10
                strengths.append(f"本業高毛利護城河 (營益率 {op_m}%)")
            elif op_m < 3.0:
                score -= 10
                warns.append(f"本業獲利空間偏薄 (營益率 {op_m}%)")
        if rev_g is not None:
            if rev_g >= 15.0:
                score += 12
                strengths.append(f"營收高速成長 ({rev_g:+.1f}%)")
            elif rev_g >= 5.0:
                score += 5
            elif rev_g <= -10.0:
                score -= 12
                warns.append(f"營收動能明顯衰退 ({rev_g:+.1f}%)")
        if pe is not None:
            if 0 < pe <= 16.0:
                score += 10
                strengths.append(f"估值具下檔防禦力 (PE {pe:.1f}x)")
            elif pe >= 38.0:
                score -= 12
                warns.append(f"本益比偏高，留意估值修正 (PE {pe:.1f}x)")
            elif pe < 0:
                score -= 10
                warns.append("本益比為負值（虧損）")
        if pb is not None and pb >= 6.0:
            score -= 5
            warns.append(f"股價淨值比偏高 (PB {pb:.1f}x)")
    except Exception:
        warns.append("部分財報欄位暫缺，採中性基準評估")

    display_name = f"{clean_id}-{stock_name}" if stock_name else clean_id

    q_score = int(np.clip(score, 10, 95))
    rating = (
        "A+ 優質核心" if q_score >= 80 else
        "A 穩健優良" if q_score >= 68 else
        "B 中性標準" if q_score >= 52 else
        "C 高估值/轉弱" if q_score >= 38 else "D 體質偏弱"
    )
    quality = FundamentalQualityReport(
        quality_score=q_score, risk_equivalent=float(100 - q_score), rating=rating,
        pe=round(pe, 1) if pe is not None else None, pb=round(pb, 2) if pb is not None else None,
        roe_pct=roe, op_margin_pct=op_m, rev_growth_pct=rev_g,
        strengths=strengths if strengths else ["財務結構大致符合產業平均水準"],
        warnings=warns,
    )

    # 個股籌碼
    frames, stk_status = [], "OK"
    df_m, st1 = _finmind_get("TaiwanStockMarginPurchaseShortSale", fm_start, end, token, data_id=clean_id)
    if st1 != "OK":
        stk_status = st1
    if not df_m.empty:
        df_m["date"] = pd.to_datetime(df_m["date"])
        df_m = df_m.set_index("date")
        m_out = pd.DataFrame(index=df_m.index)
        m_out["margin_bal"] = pd.to_numeric(df_m.get("MarginPurchaseTodayBalance"), errors="coerce")
        m_out["short_bal"] = pd.to_numeric(df_m.get("ShortSaleTodayBalance"), errors="coerce").fillna(0)
        m_limit = pd.to_numeric(df_m.get("MarginPurchaseLimit"), errors="coerce")
        m_out["margin_util_pct"] = np.where(m_limit > 0, (m_out["margin_bal"] / m_limit) * 100.0, np.nan)
        m_out["short_margin_ratio"] = np.where(
            m_out["margin_bal"] > 0, (m_out["short_bal"] / m_out["margin_bal"]) * 100.0, 0.0
        )
        frames.append(m_out)

    df_i, st2 = _finmind_get("TaiwanStockInstitutionalInvestorsBuySell", fm_start, end, token, data_id=clean_id)
    if st2 != "OK" and stk_status == "OK":
        stk_status = st2
    if not df_i.empty:
        df_i["date"] = pd.to_datetime(df_i["date"])
        df_i["net"] = (
            pd.to_numeric(df_i["buy"], errors="coerce").fillna(0)
            - pd.to_numeric(df_i["sell"], errors="coerce").fillna(0)
        )
        piv = df_i.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
        i_out = pd.DataFrame(index=piv.index)
        i_out["foreign_net"] = piv["Foreign_Investor"] if "Foreign_Investor" in piv else 0.0
        i_out["trust_net"] = piv["Investment_Trust"] if "Investment_Trust" in piv else 0.0
        i_out["smart_money_net"] = i_out["foreign_net"] + i_out["trust_net"]
        frames.append(i_out)

    stk_chips = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return stk, resolved_sym, display_name, quality, stk_chips, stk_status


# =========================================================
# v8.2 核心風控運算引擎（大盤 40% + 個股 60%、樂觀賣出價）
# =========================================================
class InstitutionalQuantRiskEngine:
    def __init__(self, config: Optional[RiskModelConfig] = None):
        self.cfg = config or RiskModelConfig()
        self.w = self.cfg.weights
        self.mri_weights = np.arange(1, self.cfg.window_days + 1, dtype=float)
        self.chip_weights = np.arange(1, self.cfg.chip_smooth_days + 1, dtype=float)

    def score_market_snapshot(self, row: pd.Series, news_risk_score: float = 50.0) -> float:
        s_us = 45.0
        sox_5d, sox_10d = row.get("sox_5d_pct", 0.0), row.get("sox_10d_pct", 0.0)
        ndx_5d, ndx_10d = row.get("ndx_5d_pct", 0.0), row.get("ndx_10d_pct", 0.0)
        if pd.notna(sox_10d):
            if sox_10d <= -8.0 or sox_5d <= -5.5:
                s_us += 38
            elif sox_10d <= -4.0 or sox_5d <= -3.0:
                s_us += 22
            elif sox_10d >= 5.0 and ndx_10d >= 2.5:
                s_us -= 20
        if pd.notna(ndx_10d):
            if ndx_10d <= -4.5 or ndx_5d <= -3.0:
                s_us += 18
            elif ndx_10d >= 3.0:
                s_us -= 8
        s_us = float(np.clip(s_us, 5, 100))

        s_rf = 42.0
        tnx, tnx_20d = row.get("tnx_close", np.nan), row.get("tnx_20d_chg", np.nan)
        twd_10d = row.get("usdtwd_10d_pct", 0.0)
        if pd.notna(tnx):
            s_rf += 35 if tnx >= 4.95 else (18 if tnx >= 4.65 else (-15 if tnx <= 4.10 else 0))
            if pd.notna(tnx_20d) and tnx_20d >= 0.30:
                s_rf += 18
        if pd.notna(twd_10d):
            s_rf += 16 if twd_10d >= 1.0 else (-12 if twd_10d <= -0.8 else 0)
        s_rf = float(np.clip(s_rf, 5, 100))

        s_tw = 42.0
        m_rank, m_5d = row.get("mkt_margin_rank_60d", np.nan), row.get("mkt_margin_5d_pct", np.nan)
        f_5d, f_10d = row.get("mkt_foreign_5d_b", np.nan), row.get("mkt_foreign_10d_b", np.nan)
        bias_60 = row.get("twii_bias_60", 0.0)

        if pd.notna(m_rank) or pd.notna(f_10d):
            if pd.notna(m_rank):
                s_tw += 18 if m_rank >= 0.88 else (9 if m_rank >= 0.78 else 0)
            if pd.notna(f_10d):
                if f_10d <= -800:
                    s_tw += 24
                elif f_10d <= -300 and (pd.isna(f_5d) or f_5d < 0):
                    s_tw += 14
                elif f_10d >= 400 or (pd.notna(f_5d) and f_5d >= 500):
                    s_tw -= 18
            if pd.notna(m_5d) and m_5d <= -3.0:
                if pd.notna(f_5d) and f_5d < 0:
                    s_tw += 16
                elif pd.notna(f_5d) and f_5d >= 150:
                    s_tw -= 14
        else:
            twii_dd, twii_hv = row.get("twii_dd_60d", 0.0), row.get("twii_hv_20", 15.0)
            if pd.notna(twii_dd) and twii_dd <= -12.0:
                s_tw += 24
            elif pd.notna(twii_dd) and twii_dd <= -6.0:
                s_tw += 12
            if pd.notna(twii_hv) and twii_hv >= 25.0:
                s_tw += 12

        if pd.notna(bias_60):
            s_tw += 14 if bias_60 >= 11.0 else (7 if bias_60 >= 7.5 else 0)
        if row.get("twii_below_ma240", 0) == 1:
            s_tw += 20
        elif row.get("twii_below_ma60", 0) == 1:
            s_tw += 11
        s_tw = float(np.clip(s_tw, 5, 100))

        s_cmd = 45.0
        oil_10d, oil_c = row.get("oil_10d_pct", 0.0), row.get("oil_close", 70.0)
        gold_10d = row.get("gold_10d_pct", 0.0)
        if pd.notna(oil_10d):
            if oil_10d >= 10.0 or (pd.notna(oil_c) and oil_c >= 92.0):
                s_cmd += 28
            elif oil_10d >= 6.0:
                s_cmd += 14
            elif oil_10d <= -14.0:
                s_cmd += 15
            elif -8.0 <= oil_10d <= -3.0:
                s_cmd -= 10
        if pd.notna(gold_10d):
            s_cmd += 18 if gold_10d >= 5.0 else (-6 if gold_10d <= -3.0 else 0)
        s_cmd = float(np.clip(s_cmd,
