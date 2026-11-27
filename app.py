import datetime
import logging
import re
import time
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
    .outlook-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 9px 10px;
        text-align: center;
    }
    .m-label { font-size: 0.72rem; color: #8b949e; }
    .m-val { font-size: 1.05rem; font-weight: 700; color: #f0f6fc; margin: 2px 0; }
    .m-note { font-size: 0.68rem; color: #8b949e; margin-bottom: 2px; }
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


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return f if np.isfinite(f) else None
    except Exception:
        return None


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
    # 第二層：個股風險 (SRI) 權重 — 大盤 40% ＋ 個股 60%（籌碼 24% / 趨勢 12% / 體質 12% / 展望 12%）
    w_mri_transmission: float = 0.400
    w_idiosyncratic: float = 0.600
    w_stock_chip_in_idio: float = 0.400
    w_stock_trend_in_idio: float = 0.200
    w_stock_health_in_idio: float = 0.200
    w_stock_outlook_in_idio: float = 0.200


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
    source_note: str = "Yahoo"
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ForwardOutlookReport:
    outlook_score: int                   # 未來展望分數 (10-95，越高越樂觀)
    risk_equivalent: float               # 換算風險分 (100 - outlook_score)
    rating: str                          # 強勁看多 / 正向偏多 / 中性觀望 / 動能轉弱 / 保守偏空
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    implied_eps_growth_pct: Optional[float] = None
    target_mean_price: Optional[float] = None
    target_upside_pct: Optional[float] = None
    analyst_count: Optional[int] = None
    rev_accel_pct: Optional[float] = None
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


def fetch_fundamentals_and_outlook_smart(
    clean_id: str, resolved_sym: str, latest_close: float, latest_dd_20d: float, token: str
) -> Tuple[FundamentalQualityReport, ForwardOutlookReport, str]:
    """
    智慧基本面＋未來展望擷取（v8.5 校準版）：
    - 修正賣方目標價樂觀偏誤，並於股價深跌時自動折價（防範滯後目標價陷阱）
    - 僅在成功抓到真實數據時快取 30 分鐘；若全為空值絕不快取
    """
    if "fund_outlook_cache" not in st.session_state:
        st.session_state["fund_outlook_cache"] = {}

    cache_key = f"{clean_id}_{bool(token)}_v85"
    cached = st.session_state["fund_outlook_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 1800):
        return cached["quality"], cached["outlook"], cached["yf_name"]

    pe = pb = roe = op_m = rev_g = None
    trailing_pe = forward_pe = target_mean = None
    analyst_cnt = None
    rev_accel = None
    yf_name = ""
    used_sources = []

    # 1) Yahoo Finance 基本面與法人前瞻資料
    try:
        info = yf.Ticker(resolved_sym).info or {}
        yf_name = info.get("shortName") or info.get("longName") or ""

        t_pe = _safe_float(info.get("trailingPE"))
        f_pe = _safe_float(info.get("forwardPE"))
        pb_raw = _safe_float(info.get("priceToBook"))
        roe_r = _safe_float(info.get("returnOnEquity"))
        op_r = _safe_float(info.get("operatingMargins"))
        rev_r = _safe_float(info.get("revenueGrowth"))
        tgt_raw = _safe_float(info.get("targetMeanPrice"))
        cnt_raw = _safe_float(info.get("numberOfAnalystOpinions"))

        trailing_pe = round(t_pe, 1) if t_pe is not None else None
        forward_pe = round(f_pe, 1) if f_pe is not None else None
        pe = trailing_pe if trailing_pe is not None else forward_pe
        pb = round(pb_raw, 2) if pb_raw is not None else None
        roe = round(roe_r * 100, 1) if roe_r is not None else None
        op_m = round(op_r * 100, 1) if op_r is not None else None
        rev_g = round(rev_r * 100, 1) if rev_r is not None else None
        target_mean = round(tgt_raw, 1) if (tgt_raw is not None and tgt_raw > 0) else None
        analyst_cnt = int(cnt_raw) if (cnt_raw is not None and cnt_raw > 0) else None

        if any(v is not None for v in (pe, pb, roe, op_m, rev_g, forward_pe, target_mean)):
            used_sources.append("Yahoo")
    except Exception:
        pass

    today = datetime.date.today()
    end_str = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # 2) FinMind TaiwanStockPER 補齊 PE / PB
    if pe is None or pb is None:
        per_start = (today - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
        df_per, st_per = _finmind_get("TaiwanStockPER", per_start, end_str, token, data_id=clean_id)
        if st_per == "OK" and not df_per.empty:
            latest_per = df_per.sort_values("date").iloc[-1]
            fm_pe = _safe_float(latest_per.get("PER"))
            fm_pb = _safe_float(latest_per.get("PBR"))
            if pe is None and fm_pe is not None and fm_pe > 0:
                pe = round(fm_pe, 1)
                if trailing_pe is None:
                    trailing_pe = pe
                if "FinMind" not in used_sources:
                    used_sources.append("FinMind")
            if pb is None and fm_pb is not None and fm_pb > 0:
                pb = round(fm_pb, 2)
                if "FinMind" not in used_sources:
                    used_sources.append("FinMind")

    # 3) FinMind TaiwanStockMonthRevenue：計算「營收動能加速度(3M vs 12M)」並補齊營收年增率
    rev_start = (today - datetime.timedelta(days=500)).strftime("%Y-%m-%d")
    df_rev, st_rev = _finmind_get("TaiwanStockMonthRevenue", rev_start, end_str, token, data_id=clean_id)
    if st_rev == "OK" and not df_rev.empty and "revenue" in df_rev.columns:
        df_rev["revenue"] = pd.to_numeric(df_rev["revenue"], errors="coerce")
        df_rev = (
            df_rev.dropna(subset=["revenue", "revenue_year", "revenue_month"])
            .sort_values(["revenue_year", "revenue_month"])
            .drop_duplicates(subset=["revenue_year", "revenue_month"], keep="last")
        )
        if len(df_rev) >= 6:
            avg_3m = float(df_rev["revenue"].tail(3).mean())
            avg_12m = float(df_rev["revenue"].tail(12).mean())
            if avg_12m > 0:
                rev_accel = round((avg_3m / avg_12m - 1.0) * 100.0, 1)
                if "FinMind" not in used_sources:
                    used_sources.append("FinMind")

        if rev_g is None and not df_rev.empty:
            last_row = df_rev.iloc[-1]
            ly_match = df_rev[
                (df_rev["revenue_year"] == int(last_row["revenue_year"]) - 1)
                & (df_rev["revenue_month"] == int(last_row["revenue_month"]))
            ]
            if not ly_match.empty:
                ly_rev = float(ly_match.iloc[-1]["revenue"])
                cur_rev = float(last_row["revenue"])
                if ly_rev > 0:
                    rev_g = round((cur_rev / ly_rev - 1.0) * 100.0, 1)
                    if "FinMind" not in used_sources:
                        used_sources.append("FinMind")

    # 4) 評分 A：歷史財務體質健檢
    q_score = 52
    q_strengths, q_warns = [], []
    if roe is not None:
        if roe >= 15.0:
            q_score += 14
            q_strengths.append(f"股東報酬優異 (ROE {roe}%)")
        elif roe >= 8.0:
            q_score += 6
            q_strengths.append(f"獲利能力穩健 (ROE {roe}%)")
        elif roe < 0:
            q_score -= 16
            q_warns.append(f"目前處於虧損狀態 (ROE {roe}%)")
    if op_m is not None:
        if op_m >= 15.0:
            q_score += 10
            q_strengths.append(f"本業高毛利護城河 (營益率 {op_m}%)")
        elif op_m < 3.0:
            q_score -= 10
            q_warns.append(f"本業獲利空間偏薄 (營益率 {op_m}%)")
    if rev_g is not None:
        if rev_g >= 15.0:
            q_score += 12
            q_strengths.append(f"營收高速成長 ({rev_g:+.1f}%)")
        elif rev_g >= 5.0:
            q_score += 5
        elif rev_g <= -10.0:
            q_score -= 12
            q_warns.append(f"營收動能明顯衰退 ({rev_g:+.1f}%)")
    if pe is not None:
        if 0 < pe <= 16.0:
            q_score += 10
            q_strengths.append(f"估值具下檔防禦力 (PE {pe:.1f}x)")
        elif pe >= 38.0:
            q_score -= 12
            q_warns.append(f"本益比偏高，留意估值修正 (PE {pe:.1f}x)")
        elif pe < 0:
            q_score -= 10
            q_warns.append("本益比為負值（虧損）")
    if pb is not None and pb >= 6.0:
        q_score -= 5
        q_warns.append(f"股價淨值比偏高 (PB {pb:.1f}x)")

    has_hist = any(v is not None for v in (pe, pb, roe, op_m, rev_g))
    if not has_hist:
        q_warns.append("暫無歷史財報數據（可能為 ETF 或資料源限流，未寫入快取）")

    q_score = int(np.clip(q_score, 10, 95))
    q_rating = (
        "A+ 優質核心" if q_score >= 80 else
        "A 穩健優良" if q_score >= 68 else
        "B 中性標準" if q_score >= 52 else
        "C 高估值/轉弱" if q_score >= 38 else "D 體質偏弱"
    )
    quality = FundamentalQualityReport(
        quality_score=q_score, risk_equivalent=float(100 - q_score), rating=q_rating,
        pe=pe, pb=pb, roe_pct=roe, op_margin_pct=op_m, rev_growth_pct=rev_g,
        source_note=" + ".join(used_sources) if used_sources else "暫無資料",
        strengths=q_strengths if q_strengths else ["財務結構大致符合產業平均水準"],
        warnings=q_warns,
    )

    # 5) 評分 B：未來展望與法人預期（嚴格校準，消除賣方樂觀通膨）
    o_score = 48.0
    o_strengths, o_warns = [], []
    implied_eps_g = None
    target_upside = None

    # 因子 1：預估獲利成長 (Trailing PE / Forward PE - 1)
    if trailing_pe is not None and forward_pe is not None and trailing_pe > 0 and forward_pe > 0:
        implied_eps_g = round((trailing_pe / forward_pe - 1.0) * 100.0, 1)
        if implied_eps_g >= 25.0:
            o_score += 12.0
            o_strengths.append(f"預估未來一年獲利高成長 ({implied_eps_g:+.1f}%，Forward PE 降至 {forward_pe}x)")
        elif implied_eps_g >= 10.0:
            o_score += 6.0
            o_strengths.append(f"預估獲利穩步擴張 ({implied_eps_g:+.1f}%)")
        elif implied_eps_g <= -10.0:
            o_score -= 14.0
            o_warns.append(f"市場預估獲利恐衰退 ({implied_eps_g:+.1f}%，Forward PE 升至 {forward_pe}x)")
        elif implied_eps_g <= -3.0:
            o_score -= 7.0
            o_warns.append(f"預估獲利微幅下修 ({implied_eps_g:+.1f}%)")

    # 因子 2：外資／分析師共識目標價空間（加入深跌折價防呆）
    if target_mean is not None and latest_close > 0:
        target_upside = round((target_mean / latest_close - 1.0) * 100.0, 1)
        cnt_txt = f"({analyst_cnt}家機構)" if analyst_cnt else ""
        is_deep_pullback = pd.notna(latest_dd_20d) and latest_dd_20d <= -12.0

        if target_upside >= 30.0:
            bonus = 6.0 if is_deep_pullback else 12.0
            o_score += bonus
            o_strengths.append(f"法人共識目標價 {target_mean:.1f} 元{cnt_txt}，潛在空間達 {target_upside:+.1f}%")
            if is_deep_pullback:
                o_warns.append(f"股價近月回檔 {latest_dd_20d:.1f}%，外資目標價可能尚未反應下修（已自動折價）")
        elif target_upside >= 15.0:
            bonus = 3.0 if is_deep_pullback else 6.0
            o_score += bonus
            o_strengths.append(f"法人目標價 {target_mean:.1f} 元{cnt_txt}，尚有 {target_upside:+.1f}% 空間")
            if is_deep_pullback:
                o_warns.append(f"股價近月回檔 {latest_dd_20d:.1f}%，留意法人目標價滯後風險")
        elif target_upside <= -8.0:
            o_score -= 14.0
            o_warns.append(f"現價已明顯高於法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)，留意追高")
        elif target_upside <= -2.0:
            o_score -= 7.0
            o_warns.append(f"現價逼近或略高於法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)")

    # 因子 3：營收動能加速度（近 3 月均營收 vs 近 12 月均營收，硬數據給較高權重）
    if rev_accel is not None:
        if rev_accel >= 18.0:
            o_score += 14.0
            o_strengths.append(f"近 3 月營收動能顯著加速 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel >= 6.0:
            o_score += 7.0
            o_strengths.append(f"近 3 月營收溫和增溫 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel <= -12.0:
            o_score -= 14.0
            o_warns.append(f"近 3 月營收動能明顯鈍化 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel <= -4.0:
            o_score -= 7.0
            o_warns.append(f"近 3 月營收略低於年均 ({rev_accel:+.1f}%)")

    has_outlook = any(v is not None for v in (implied_eps_g, target_upside, rev_accel))
    if not has_outlook:
        o_score = 50.0
        o_warns.append("目前缺乏外資預估與月營收趨勢資料，展望採中性 50 分基準")

    o_score_int = int(np.clip(round(o_score), 10, 95))
    o_rating = (
        "強勁看多" if o_score_int >= 78 else
        "正向偏多" if o_score_int >= 62 else
        "中性觀望" if o_score_int >= 45 else
        "動能轉弱" if o_score_int >= 32 else "保守偏空"
    )
    outlook = ForwardOutlookReport(
        outlook_score=o_score_
