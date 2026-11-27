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

# 關閉非關鍵警告與 yfinance 冗餘日誌，保持 Streamlit 終端機與畫面乾淨
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =========================================================
# 頁面基本設定與黑底質感 CSS（手機優先排版）
# =========================================================
st.set_page_config(
    page_title="台股量化風控儀表板 v10.1",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def _render_html(html: str):
    """移除每行前導空白與空行，徹底避免 Streamlit Markdown 將 HTML 誤判為縮排程式碼區塊"""
    clean = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


# 自訂深色主題樣式：風險卡片、狀態徽章、四象限標籤、指標網格與免責聲明
_render_html("""
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
    .card-value { font-size: 1.80rem; font-weight: 700; line-height: 1.2; }
    .card-sub { font-size: 0.82rem; color: #8b949e; margin-top: 6px; line-height: 1.5; }
    .badge-red { background: rgba(255,123,114,0.18); color: #ff7b72; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin: 2px 4px 2px 0; }
    .badge-yellow { background: rgba(210,153,34,0.20); color: #e3b341; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin: 2px 4px 2px 0; }
    .badge-green { background: rgba(63,185,80,0.18); color: #3fb950; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin: 2px 4px 2px 0; }
    .badge-blue { background: rgba(88,166,255,0.18); color: #58a6ff; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin: 2px 4px 2px 0; }
    .badge-purple { background: rgba(210,168,255,0.18); color: #d2a8ff; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin: 2px 4px 2px 0; }
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
    """)

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
    """優先讀取側邊欄手動輸入的 Token，其次讀取 Streamlit Secrets 中的 FINMIND_TOKEN"""
    if st.session_state.get("user_finmind_token"):
        return st.session_state["user_finmind_token"].strip()
    try:
        return st.secrets.get("FINMIND_TOKEN", "").strip()
    except Exception:
        return ""


def _safe_float(val) -> Optional[float]:
    """安全轉換為有限浮點數，遇到 None、NaN 或無窮大時回傳 None"""
    try:
        if val is None:
            return None
        f = float(val)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def round_tw_tick(price: float) -> float:
    """依台股證券交易升降單位規則將價格四捨五入至合法跳動單位"""
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
    """格式化數值顯示，空值統一回傳 N/A"""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:+.1f}{suffix}" if signed else f"{val:.1f}{suffix}"


def _latest_rolling_zscore(series: pd.Series, window: int = 60, min_periods: int = 20) -> Optional[float]:
    """計算序列最新值相對於過去 window 日自身分佈的標準化 Z-Score（自適應動態門檻核心）"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < min_periods:
        return None
    tail = s.tail(window)
    std = float(tail.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        return 0.0
    z = (float(tail.iloc[-1]) - float(tail.mean())) / std
    return round(float(np.clip(z, -4.0, 4.0)), 2)


def get_traffic_light(score: int) -> Tuple[str, str, str, str]:
    """依風險分數（0~100）回傳 (燈號圖示, 狀態文字, 文字色碼, 徽章 CSS 類別)"""
    if score >= 65:
        return "🔴", "危險", "#ff7b72", "badge-red"
    elif score >= 48:
        return "🟡", "中等", "#e3b341", "badge-yellow"
    return "🟢", "安全", "#3fb950", "badge-green"


@dataclass
class GoldenRatioWeights:
    """多因子權重設定：第一層大盤系統風險 (MRI) 與第二層個股風險 (SRI) 基準權重"""
    # 第一層：大盤系統風險 (MRI) 六大維度權重（總和為 1.0）
    w_us_tech: float = 0.200       # 美股科技/半導體動能
    w_rates_fx: float = 0.180      # 美債殖利率與新台幣匯率
    w_tw_market: float = 0.238     # 台股大盤位階、融資與法人動向
    w_commodities: float = 0.120   # 原油與黃金通膨/避險訊號
    w_btc_liquidity: float = 0.100 # 比特幣流動性風險偏好
    w_vix_news: float = 0.162      # VIX 恐慌指數與新聞情緒
    # 第二層：個股風險 (SRI) 基準權重（實際會由「動態狀態機 × Rolling Beta」自動切換）
    w_mri_transmission: float = 0.400
    w_idiosyncratic: float = 0.600
    w_stock_chip_in_idio: float = 0.500
    w_stock_health_in_idio: float = 0.250
    w_stock_outlook_in_idio: float = 0.250


@dataclass
class RiskModelConfig:
    """風控模型視窗與高風險鎖定地板參數"""
    window_days: int = 10
    chip_smooth_days: int = 5
    weights: GoldenRatioWeights = field(default_factory=GoldenRatioWeights)
    lock_floor_level1: float = 68.0  # 符合 2 項系統性惡化條件時的大盤風險地板
    lock_floor_level2: float = 78.0  # 符合 3 項系統性惡化條件時的大盤風險地板


@dataclass
class FundamentalQualityReport:
    """歷史財務體質評估報告（分數越高代表體質越佳，risk_equivalent 為對應風險分）"""
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
    """未來展望與法人預期報告（outlook_score 越高越樂觀，risk_equivalent = 100 - outlook_score）"""
    outlook_score: int
    risk_equivalent: float
    rating: str
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    implied_eps_growth_pct: Optional[float] = None
    target_mean_price: Optional[float] = None
    target_upside_pct: Optional[float] = None
    analyst_count: Optional[int] = None
    rev_accel_pct: Optional[float] = None
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ChipPainReport:
    """自適應籌碼痛點指數報告：結合 20 日 VWAP 乖離、20 日融資變化與 60 日自適應 Z-Score"""
    status: str                          # 多殺多預警 / 籌碼乾淨 / 中性正常 / 暫無融資資料
    risk_adj: float                      # 對籌碼風險分的加減分 (-10 ~ +15)
    vwap_20: Optional[float] = None
    price_vs_vwap_pct: Optional[float] = None
    margin_chg_20d_pct: Optional[float] = None
    vwap_zscore_60d: Optional[float] = None
    margin_zscore_60d: Optional[float] = None
    note: str = ""


@dataclass
class DownsideAlphaReport:
    """逆風韌性係數報告：統計大盤單日跌逾 1% 時的個股表現與法人買超"""
    has_badge: bool                      # 是否獲得「🛡️ 逆風抗跌徽章」
    down_days_count: int = 0
    avg_stock_ret_pct: Optional[float] = None
    avg_twii_ret_pct: Optional[float] = None
    avg_inst_net_lots: Optional[float] = None
    note: str = ""


@dataclass
class DynamicFactorReport:
    """底層動態化因子報告：Rolling Beta 權重調校、SRI 風險加速度、自適應 Z-Score 狀態"""
    rolling_beta_60d: float
    base_w_mri: float
    beta_adj_w_mri: float
    sri_velocity_3d: float               # 近 3 日 SRI 風險斜率（正值=升溫，負值=收斂）
    velocity_tag: str                    # 加速惡化 / 緩步升溫 / 風險收斂 / 平穩
    velocity_adj: float                  # 套用於最終 SRI 的加速度微調分
    atr_norm_dd: Optional[float] = None  # 以個股自身 ATR 正規化之 20 日回撤倍數
    note: str = ""


@dataclass
class ActiveAlphaReport:
    """主動式進攻因子報告與四象限決策標籤（解決死魚股拿高安全分盲點）"""
    alpha_score: int                     # 5 ~ 98，越高代表資金主動表態越強
    rating: str                          # 強力點火 / 主動偏多 / 溫和蓄勢 / 動能偏弱 / 資金退潮
    quadrant: str                        # 🚀 主升段首選 / 🛡️ 防禦盤整 / ⚡ 高檔博弈 / 🚫 弱勢迴避
    quadrant_badge_cls: str              # badge-green / badge-blue / badge-yellow / badge-red
    quadrant_desc: str
    rs_vs_ma20_pct: Optional[float] = None
    rs_new_high_60d: bool = False
    vol_squeeze_ratio: Optional[float] = None
    is_squeeze_breakout: bool = False
    stealth_accum_pct: Optional[float] = None
    is_stealth_accum: bool = False
    signals: List[str] = field(default_factory=list)


@dataclass
class AdvancedRiskOverlay:
    """v10 進階風控與主動因子彙整結果，供單股診斷與批量掃描呈現"""
    regime_name: str
    w_mri_dynamic: float
    w_idio_dynamic: float
    divergence_penalty: float
    divergence_warning: Optional[str]
    pain_report: ChipPainReport
    downside_alpha: DownsideAlphaReport
    dynamic_factors: DynamicFactorReport
    active_alpha: ActiveAlphaReport


# =========================================================
# 進階動態與主動量化因子演算法（v10.0 核心升級）
# =========================================================
def compute_rolling_beta(
    stock_close: pd.Series,
    twii_close: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> float:
    """
    計算個股近 60 日相對於加權指數的 Rolling Beta（滾動敏感度）：
    Beta = Cov(r_stock, r_mkt) / Var(r_mkt)
    """
    s_ret = stock_close.pct_change()
    m_ret = twii_close.pct_change()
    aligned = pd.concat([s_ret.rename("s"), m_ret.rename("m")], axis=1).dropna().tail(window)
    if len(aligned) < min_periods:
        return 1.0
    var_m = float(aligned["m"].var(ddof=0))
    if not np.isfinite(var_m) or var_m < 1e-10:
        return 1.0
    cov_sm = float(np.cov(aligned["s"], aligned["m"], ddof=0)[0, 1])
    beta = cov_sm / var_m
    if not np.isfinite(beta):
        return 1.0
    return round(float(np.clip(beta, 0.25, 2.20)), 2)


def get_dynamic_regime_weights(mri_score: float, rolling_beta: float = 1.0) -> Tuple[float, float, float, str]:
    """
    1. 動態權重狀態機 × 個股 Rolling Beta (Adaptive Regime-Switching)：
    - 先依大盤 MRI 決定基準權重（低波動 15% / 常態 40% / 恐慌 65%）
    - 再乘上個股 60 日 Beta 敏感度係數（限制在 0.55x ~ 1.45x），並將最終大盤權重收斂於 8% ~ 78%
    回傳：(最終大盤權重, 最終個股權重, 基準大盤權重, 狀態名稱)
    """
    if mri_score < 45.0:
        base_mri, regime = 0.15, "低波動多頭期"
    elif mri_score >= 70.0:
        base_mri, regime = 0.65, "系統性恐慌期"
    else:
        base_mri, regime = 0.40, "常態震盪期"

    beta_mult = float(np.clip(rolling_beta, 0.55, 1.45))
    w_mri_adj = float(np.clip(base_mri * beta_mult, 0.08, 0.78))
    w_idio_adj = round(1.0 - w_mri_adj, 4)
    w_mri_adj = round(w_mri_adj, 4)
    label = f"{regime} ({int(round(w_mri_adj * 100))}/{int(round(w_idio_adj * 100))}｜β={rolling_beta:.2f})"
    return w_mri_adj, w_idio_adj, base_mri, label


def detect_divergence_penalty(
    quality_risk: float,
    outlook_risk: float,
    chip_risk: float,
    trend_risk: float,
) -> Tuple[float, Optional[str]]:
    """
    2. 訊號背離非線性陷阱偵測 (Divergence Penalty)：
    當基本面/展望風險極低（<= 35，代表基本面佳）但籌碼/趨勢風險極高（>= 72，代表法人撤出或破線），
    避免線性平均掩蓋「利多出貨」陷阱，直接加入 +15 ~ +20 風險懲罰與警語。
    """
    fund_avg_risk = (quality_risk + outlook_risk) / 2.0
    action_avg_risk = (chip_risk + trend_risk) / 2.0
    if fund_avg_risk <= 35.0 and action_avg_risk >= 72.0:
        gap = action_avg_risk - fund_avg_risk
        penalty = float(np.clip(15.0 + (gap - 37.0) * 0.2, 15.0, 20.0))
        warning = "⚠️ 偵測到利多背離：基本面佳但法人異常撤出，防範提前反應利空"
        return round(penalty, 1), warning
    return 0.0, None


def compute_chip_pain_index(
    stock_df: pd.DataFrame,
    margin_series: Optional[pd.Series] = None,
) -> ChipPainReport:
    """
    3. 自適應籌碼痛點指數 (Adaptive Pain Index)：
    同時計算「絕對門檻」與「過去 60 日自身 Z-Score」：
    - 解決低波動股（如金融股）跌 4% 其實已是 -2σ 異常、高波動股跌 4% 僅屬日常雜訊的問題
    """
    if stock_df is None or len(stock_df) < 20:
        return ChipPainReport(status="資料不足", risk_adj=0.0, note="交易日不足 20 天")

    tp = (stock_df["High"] + stock_df["Low"] + stock_df["Close"]) / 3.0
    vol = stock_df["Volume"].fillna(0.0)
    roll_vwap = (tp * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    roll_vwap = roll_vwap.fillna(stock_df["Close"].rolling(20).mean())

    vwap_20 = float(roll_vwap.iloc[-1])
    latest_close = float(stock_df["Close"].iloc[-1])
    vwap_gap_series = (stock_df["Close"] / roll_vwap - 1.0) * 100.0
    price_vs_vwap_pct = round(float(vwap_gap_series.iloc[-1]), 2) if np.isfinite(vwap_gap_series.iloc[-1]) else None
    vwap_z = _latest_rolling_zscore(vwap_gap_series, window=60, min_periods=20)

    margin_chg_pct = None
    margin_z = None
    if margin_series is not None:
        m = pd.to_numeric(margin_series, errors="coerce").dropna()
        if len(m) >= 20 and m.iloc[-20] > 0:
            m_chg_series = m.pct_change(20) * 100.0
            margin_chg_pct = round(float(m_chg_series.iloc[-1]), 1)
            margin_z = _latest_rolling_zscore(m_chg_series, window=60, min_periods=20)

    z_txt = f"（乖離 Z={vwap_z:+.1f}σ" + (f" / 融資 Z={margin_z:+.1f}σ）" if margin_z is not None else "）") if vwap_z is not None else ""

    if price_vs_vwap_pct is None or margin_chg_pct is None:
        return ChipPainReport(
            status="暫無融資資料",
            risk_adj=0.0,
            vwap_20=round(vwap_20, 2),
            price_vs_vwap_pct=price_vs_vwap_pct,
            vwap_zscore_60d=vwap_z,
            note=f"無法取得完整 20 日融資餘額；現價距20日均價 {_fmt_num(price_vs_vwap_pct, '%', signed=True)}{z_txt}",
        )

    # 多殺多預警：符合傳統絕對門檻 OR 自適應雙邊異常（融資激增 >= +1.5σ 且跌破均價 <= -1.5σ）
    is_classic_squeeze = (margin_chg_pct >= 10.0 and price_vs_vwap_pct <= -8.0)
    is_adaptive_squeeze = (
        margin_z is not None and vwap_z is not None
        and margin_z >= 1.5 and vwap_z <= -1.5 and price_vs_vwap_pct <= -2.5
    )
    if is_classic_squeeze or is_adaptive_squeeze:
        trigger_lbl = "自適應異常" if (is_adaptive_squeeze and not is_classic_squeeze) else "槓桿套牢"
        return ChipPainReport(
            status="多殺多預警",
            risk_adj=15.0,
            vwap_20=round(vwap_20, 2),
            price_vs_vwap_pct=price_vs_vwap_pct,
            margin_chg_20d_pct=margin_chg_pct,
            vwap_zscore_60d=vwap_z,
            margin_zscore_60d=margin_z,
            note=f"【{trigger_lbl}】近20日融資 {margin_chg_pct:+.1f}%，現價低於20日均價 {price_vs_vwap_pct:.1f}%{z_txt}，留意斷頭賣壓",
        )

    # 籌碼乾淨度：符合傳統絕對門檻 OR 自適應沉澱（融資顯著收斂 <= -1.0σ 且站穩均價之上 >= +0.5σ）
    is_classic_clean = (margin_chg_pct <= -5.0 and price_vs_vwap_pct >= 0.0)
    is_adaptive_clean = (
        margin_z is not None and vwap_z is not None
        and margin_z <= -1.0 and vwap_z >= 0.5 and price_vs_vwap_pct >= 0.0
    )
    if is_classic_clean or is_adaptive_clean:
        return ChipPainReport(
            status="籌碼乾淨",
            risk_adj=-10.0,
            vwap_20=round(vwap_20, 2),
            price_vs_vwap_pct=price_vs_vwap_pct,
            margin_chg_20d_pct=margin_chg_pct,
            vwap_zscore_60d=vwap_z,
            margin_zscore_60d=margin_z,
            note=f"近20日融資減少 {margin_chg_pct:+.1f}% 且站穩20日均價 ({price_vs_vwap_pct:+.1f}%){z_txt}，下檔支撐強",
        )

    return ChipPainReport(
        status="中性正常",
        risk_adj=0.0,
        vwap_20=round(vwap_20, 2),
        price_vs_vwap_pct=price_vs_vwap_pct,
        margin_chg_20d_pct=margin_chg_pct,
        vwap_zscore_60d=vwap_z,
        margin_zscore_60d=margin_z,
        note=f"現價距20日均價 {price_vs_vwap_pct:+.1f}%，近20日融資變動 {margin_chg_pct:+.1f}%{z_txt}",
    )


def compute_downside_alpha(
    stock_df: pd.DataFrame,
    twii_df: pd.DataFrame,
    inst_net_shares_series: Optional[pd.Series] = None,
    lookback_days: int = 60,
) -> DownsideAlphaReport:
    """
    4. 逆風韌性係數 (Downside Alpha)：
    掃描近 60 個交易日中「加權指數單日跌幅 <= -1%」的日子，計算個股平均報酬與法人淨買超張數。
    """
    if stock_df is None or twii_df is None or stock_df.empty or twii_df.empty:
        return DownsideAlphaReport(has_badge=False, note="大盤或個股歷史價格不足")
    s_ret = stock_df["Close"].pct_change() * 100.0
    m_ret = twii_df["Close"].pct_change() * 100.0
    aligned = pd.concat([s_ret.rename("stock"), m_ret.rename("twii")], axis=1).dropna().tail(lookback_days)
    down_days = aligned[aligned["twii"] <= -1.0]
    cnt = len(down_days)
    if cnt == 0:
        return DownsideAlphaReport(has_badge=False, down_days_count=0, note="近 60 日無大盤單日跌逾 1% 之交易日")
    avg_s = round(float(down_days["stock"].mean()), 2)
    avg_m = round(float(down_days["twii"].mean()), 2)
    avg_inst_lots = None
    if inst_net_shares_series is not None and not inst_net_shares_series.empty:
        inst_aligned = inst_net_shares_series.reindex(down_days.index).dropna()
        if not inst_aligned.empty:
            avg_inst_lots = round(float(inst_aligned.mean()) / 1000.0, 1)
    is_resilient = cnt >= 2 and (
        avg_s > 0.0
        or (avg_inst_lots is not None and avg_inst_lots > 0.0 and (avg_s - avg_m) >= 1.0)
    )
    inst_txt = f"，法人逆風日均買超 {avg_inst_lots:+.0f} 張" if avg_inst_lots is not None else ""
    note = f"近60日大盤大跌共 {cnt} 天（均跌 {avg_m:.2f}%），本股均表現 {avg_s:+.2f}%{inst_txt}"
    return DownsideAlphaReport(
        has_badge=is_resilient,
        down_days_count=cnt,
        avg_stock_ret_pct=avg_s,
        avg_twii_ret_pct=avg_m,
        avg_inst_net_lots=avg_inst_lots,
        note=note,
    )


def evaluate_sri_velocity(base_sri_history: List[float]) -> Tuple[float, str, float]:
    """
    5. 風險加速度 (SRI Velocity / 二階斜率)：
    追蹤近 3 個交易日的風險變化速度：
    - 急升 (>= +10 分)：判定「加速惡化」，提前加罰 +3 ~ +6 分預警
    - 緩升 (>= +5 分)：判定「緩步升溫」，加罰 +2 分
    - 急降 (<= -8 分)：判定「風險收斂」，給予 -3 分正向轉折緩解
    """
    if len(base_sri_history) < 4:
        return 0.0, "平穩", 0.0
    vel_3d = round(float(base_sri_history[-1] - base_sri_history[-4]), 1)
    if vel_3d >= 10.0:
        adj = round(float(np.clip(3.0 + (vel_3d - 10.0) * 0.4, 3.0, 6.0)), 1)
        return vel_3d, "加速惡化", adj
    elif vel_3d >= 5.0:
        return vel_3d, "緩步升溫", 2.0
    elif vel_3d <= -8.0:
        return vel_3d, "風險收斂", -3.0
    return vel_3d, "平穩", 0.0


def compute_active_alpha_and_quadrant(
    df: pd.DataFrame,
    latest_sri: int,
    outlook_score: int,
    downside_report: DownsideAlphaReport,
) -> ActiveAlphaReport:
    """
    6. 主動式進攻因子 (Active Alpha Score) 與四象限分類：
    三大主動訊號：
    (A) 相對大盤強弱動能 (RS Momentum)：RS 比值距 20 日均線乖離、是否創 60 日新高
    (B) 波動率壓縮突破 (Volatility Squeeze & Breakout)：ATR 10/60 壓縮後帶量站上 VWAP
    (C) 法人低檔吸納與二階轉折 (Stealth Accumulation & Flow Inflection)：股價未大漲時法人連買，或由賣轉買
    """
    if df is None or len(df) < 25:
        return ActiveAlphaReport(
            alpha_score=50, rating="溫和蓄勢", quadrant="🛡️ 防禦盤整",
            quadrant_badge_cls="badge-blue", quadrant_desc="歷史交易日不足，暫以中性基準顯示",
            signals=["交易日資料尚不足以計算完整主動動能"],
        )

    last = df.iloc[-1]
    score = 45.0
    signals: List[str] = []

    # (A) 相對強弱動能 (RS = 個股 / 加權指數)
    rs_vs_ma20 = _safe_float(last.get("rs_vs_ma20_pct"))
    rs_new_high = bool(last.get("rs_new_high_60d", 0) == 1)
    if rs_new_high:
        score += 16.0
        signals.append("相對大盤強弱 (RS) 創近 60 日新高，資金主動集中")
    elif rs_vs_ma20 is not None:
        if rs_vs_ma20 >= 2.5:
            score += 10.0
            signals.append(f"相對大盤走強（RS 領先月均 {rs_vs_ma20:+.1f}%）")
        elif rs_vs_ma20 <= -3.5:
            score -= 12.0
            signals.append(f"相對大盤明顯落後（RS 低於月均 {rs_vs_ma20:+.1f}%）")
        elif rs_vs_ma20 < -1.0:
            score -= 5.0

    # (B) 波動率壓縮與帶量點火 (ATR 10 / ATR 60)
    sq_ratio = _safe_float(last.get("vol_squeeze_ratio"))
    min_sq_5d = _safe_float(last.get("min_sq_5d"))
    vol_ratio = _safe_float(last.get("stk_vol_ratio_20"))
    close_p = _safe_float(last.get("known_stk_close")) or 0.0
    vwap_20 = _safe_float(last.get("stk_vwap_20")) or close_p
    ret_10d = _safe_float(last.get("stk_ret_10d")) or 0.0
    sm_5d = _safe_float(last.get("smart_5d_vol_ratio"))
    sm_prev5d = _safe_float(last.get("smart_prev5d_vol_ratio"))
    cmf_10 = _safe_float(last.get("stk_cmf_10")) or 0.0

    is_squeeze_breakout = False
    has_buyer_support = (sm_5d is not None and sm_5d > 0.0) or (sm_5d is None and cmf_10 >= 0.05)

    if (
        min_sq_5d is not None and min_sq_5d <= 0.85
        and close_p > vwap_20
        and vol_ratio is not None and vol_ratio >= 1.25
        and has_buyer_support
    ):
        is_squeeze_breakout = True
        score += 16.0
        signals.append(f"波動率壓縮後帶量突破（量比 {vol_ratio:.1f}x，主力主動點火）")
    elif sq_ratio is not None and sq_ratio <= 0.80 and close_p >= vwap_20 * 0.99:
        score += 8.0
        signals.append(f"波動率高度壓縮蓄勢中（ATR壓縮比 {sq_ratio:.2f}，待量能表態）")
    elif vol_ratio is not None and vol_ratio < 0.65 and abs(ret_10d) < 2.5:
        score -= 9.0
        signals.append("近期量縮橫盤、缺乏資金表態（防範死魚盤）")

    # (C) 法人低檔吸納度 & 二階賣壓轉折
    is_stealth_accum = False
    if sm_5d is not None:
        if sm_5d >= 3.0 and -3.5 <= ret_10d <= 6.5:
            is_stealth_accum = True
            score += 15.0
            signals.append(f"法人低檔溫和吸納（5日買超佔量 {sm_5d:+.1f}%，股價尚未急漲）")
        elif sm_5d >= 4.0 and ret_10d > 6.5:
            score += 9.0
            signals.append(f"法人順勢推升動能（5日買超佔量 {sm_5d:+.1f}%）")
        elif sm_prev5d is not None and sm_prev5d <= -2.0 and sm_5d > 0.5:
            score += 11.0
            signals.append("法人籌碼由賣轉買（二階正向轉折訊號）")
        elif sm_5d <= -3.0:
            score -= 12.0
            signals.append(f"法人近 5 日呈現明顯調節（佔量 {sm_5d:+.1f}%）")
    else:
        # 無法人明細時改以資金流量 CMF 代理主動買盤
        if cmf_10 >= 0.14:
            score += 10.0
            signals.append("價量資金流向 (CMF) 呈現主動淨流入")
        elif cmf_10 <= -0.14:
            score -= 10.0
            signals.append("價量資金流向 (CMF) 呈現資金流出")

    # (D) 未來展望與逆風韌性共振加成
    if outlook_score >= 65:
        score += 6.0
    if downside_report.has_badge:
        score += 6.0
        signals.append("具備大盤逆風抗跌護盤屬性，攻守共振")

    alpha_int = int(np.clip(round(score), 5, 98))
    alpha_rating = (
        "強力點火" if alpha_int >= 76 else
        "主動偏多" if alpha_int >= 60 else
        "溫和蓄勢" if alpha_int >= 46 else
        "動能偏弱" if alpha_int >= 32 else "資金退潮"
    )

    # 四象限決策判定（橫軸：主動 Alpha 門檻 60；縱軸：風控 SRI 門檻 52）
    low_risk = latest_sri < 52
    high_alpha = alpha_int >= 60
    if low_risk and high_alpha:
        quad, q_cls, q_desc = "🚀 主升段首選", "badge-green", "風險可控且主動資金點火，攻守兼備首選"
    elif low_risk and not high_alpha:
        quad, q_cls, q_desc = "🛡️ 防禦盤整", "badge-blue", "下檔風險低但缺乏主動催化，屬收息防禦或量縮整理"
    elif not low_risk and high_alpha:
        quad, q_cls, q_desc = "⚡ 高檔博弈", "badge-yellow", "主動動能強但風控偏高，適合嚴守防守價短打"
    else:
        quad, q_cls, q_desc = "🚫 弱勢迴避", "badge-red", "高風險且資金退潮，建議保守觀望避開"

    if not signals:
        signals.append("目前資金動能與波動率皆處於常態中性區間")

    return ActiveAlphaReport(
        alpha_score=alpha_int,
        rating=alpha_rating,
        quadrant=quad,
        quadrant_badge_cls=q_cls,
        quadrant_desc=q_desc,
        rs_vs_ma20_pct=round(rs_vs_ma20, 2) if rs_vs_ma20 is not None else None,
        rs_new_high_60d=rs_new_high,
        vol_squeeze_ratio=round(sq_ratio, 2) if sq_ratio is not None else None,
        is_squeeze_breakout=is_squeeze_breakout,
        stealth_accum_pct=round(sm_5d, 1) if sm_5d is not None else None,
        is_stealth_accum=is_stealth_accum,
        signals=signals,
    )


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
    """抓取 Google 新聞 RSS 標題並計算消息面風險分數（20~85，50 為中性）"""
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
    """封裝 yfinance 歷史行情擷取並統一時區為無時區日期索引"""
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
    """封裝 FinMind v4 REST API 查詢並回傳 (DataFrame, 狀態訊息)"""
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
    """快取全市場共用總經指標、加權指數與大盤融資/外資籌碼（每小時更新）"""
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
    智慧基本面＋未來展望擷取（Yahoo Finance + FinMind 雙重備援）：
    - 修正賣方目標價樂觀偏誤，股價深跌時自動折價防範滯後陷阱
    - 僅在成功抓到真實數據時寫入 session_state 快取 30 分鐘；若全為空值則不快取
    """
    if "fund_outlook_cache" not in st.session_state:
        st.session_state["fund_outlook_cache"] = {}
    cache_key = f"{clean_id}_{bool(token)}_v101"
    cached = st.session_state["fund_outlook_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 1800):
        return cached["quality"], cached["outlook"], cached["yf_name"]

    pe = pb = roe = op_m = rev_g = None
    trailing_pe = forward_pe = target_mean = None
    analyst_cnt = None
    rev_accel = None
    yf_name = ""
    used_sources = []

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

    # 評分 A：歷史財務體質健檢
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

    # 評分 B：未來展望與法人預期（v10.1 嚴格校準：修正賣方樂觀偏誤、極端值封頂與深跌現實鎖定）
    o_score = 45.0
    o_strengths, o_warns = [], []
    implied_eps_g = None
    target_upside = None
    is_deep_pullback = pd.notna(latest_dd_20d) and latest_dd_20d <= -12.0
    is_severe_crash = pd.notna(latest_dd_20d) and latest_dd_20d <= -15.0
    strong_eps = strong_tgt = strong_rev = False

    # 因子 1：預估獲利成長 (Trailing PE / Forward PE - 1)
    if trailing_pe is not None and forward_pe is not None and trailing_pe > 0 and forward_pe > 0:
        implied_eps_g = round((trailing_pe / forward_pe - 1.0) * 100.0, 1)
        if implied_eps_g > 150.0:
            o_score += 8.0
            strong_eps = True
            o_strengths.append(f"預估未來一年獲利大幅跳升 ({implied_eps_g:+.1f}%，Forward PE 降至 {forward_pe}x)")
            o_warns.append(f"預估獲利增幅極端 ({implied_eps_g:+.1f}%)，可能受低基期或預估口徑差異影響（已限制加分上限）")
        elif implied_eps_g >= 30.0:
            o_score += 11.0
            strong_eps = True
            o_strengths.append(f"預估未來一年獲利高成長 ({implied_eps_g:+.1f}%，Forward PE 降至 {forward_pe}x)")
        elif implied_eps_g >= 15.0:
            o_score += 6.0
            strong_eps = True
            o_strengths.append(f"預估獲利穩步擴張 ({implied_eps_g:+.1f}%)")
        elif implied_eps_g >= 5.0:
            o_score += 2.0
        elif implied_eps_g <= -10.0:
            o_score -= 14.0
            o_warns.append(f"市場預估獲利恐衰退 ({implied_eps_g:+.1f}%，Forward PE 升至 {forward_pe}x)")
        elif implied_eps_g <= -3.0:
            o_score -= 7.0
            o_warns.append(f"預估獲利微幅下修 ({implied_eps_g:+.1f}%)")

    # 因子 2：外資／分析師共識目標價空間（扣除賣方常態樂觀偏誤，深跌時完全不予加分）
    if target_mean is not None and latest_close > 0:
        target_upside = round((target_mean / latest_close - 1.0) * 100.0, 1)
        cnt_txt = f"({analyst_cnt}家機構)" if analyst_cnt else ""
        if target_upside >= 10.0 and is_deep_pullback:
            o_warns.append(
                f"法人目標價 {target_mean:.1f} 元{cnt_txt} 雖顯示 {target_upside:+.1f}% 空間，但近月已回檔 {latest_dd_20d:.1f}%，防範滯後下修（暫不計入加分）"
            )
        elif target_upside > 80.0:
            o_score += 6.0
            strong_tgt = True
            o_strengths.append(f"法人共識目標價 {target_mean:.1f} 元{cnt_txt}，潛在空間 {target_upside:+.1f}%")
            o_warns.append("目標價溢價空間異常偏高（>80%），留意報告過時或極端預估（已限制加分）")
        elif target_upside >= 30.0:
            o_score += 10.0
            strong_tgt = True
            o_strengths.append(f"法人共識目標價 {target_mean:.1f} 元{cnt_txt}，潛在空間達 {target_upside:+.1f}%")
        elif target_upside >= 18.0:
            o_score += 5.0
            strong_tgt = True
            o_strengths.append(f"法人目標價 {target_mean:.1f} 元{cnt_txt}，尚有 {target_upside:+.1f}% 空間")
        elif target_upside >= 10.0:
            o_score += 2.0
        elif target_upside <= -8.0:
            o_score -= 14.0
            o_warns.append(f"現價已明顯高於法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)，留意追高")
        elif target_upside <= -2.0:
            o_score -= 7.0
            o_warns.append(f"現價逼近法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)")

    # 因子 3：營收動能加速度（近 3 月均營收 vs 近 12 月均營收，硬數據維持較高權重）
    if rev_accel is not None:
        if rev_accel >= 20.0:
            o_score += 13.0
            strong_rev = True
            o_strengths.append(f"近 3 月營收動能顯著加速 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel >= 10.0:
            o_score += 7.0
            strong_rev = True
            o_strengths.append(f"近 3 月營收溫和增溫 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel >= 4.0:
            o_score += 3.0
        elif rev_accel <= -12.0:
            o_score -= 14.0
            o_warns.append(f"近 3 月營收動能明顯鈍化 (較年均 {rev_accel:+.1f}%)")
        elif rev_accel <= -4.0:
            o_score -= 7.0
            o_warns.append(f"近 3 月營收略低於年均 ({rev_accel:+.1f}%)")

    # 三因子健康共振加成：唯有三項指標同步達標且股價未深跌才給予共振分數
    if strong_eps and strong_tgt and strong_rev and pd.notna(latest_dd_20d) and latest_dd_20d > -8.0:
        o_score += 6.0
        o_strengths.append("獲利預估、法人目標與營收加速三因子健康共振")

    # 價格重挫現實校準：當股價急殺時，強制壓抑展望分數避免與盤面嚴重脫節
    if is_severe_crash:
        o_score = min(o_score - 10.0, 64.0)
        o_warns.append(f"近月重挫 {latest_dd_20d:.1f}%，價格走勢與基本面預期背離，已強制壓抑展望上限（止穩前不評偏多）")
    elif is_deep_pullback:
        o_score = min(o_score - 5.0, 76.0)
        o_warns.append(f"近月回檔 {latest_dd_20d:.1f}%，趨勢轉弱期間已限縮展望最高評級")

    has_outlook = any(v is not None for v in (implied_eps_g, target_upside, rev_accel))
    if not has_outlook:
        o_score = 50.0
        o_warns.append("目前缺乏外資預估與月營收趨勢資料，展望採中性 50 分基準")

    o_score_int = int(np.clip(round(o_score), 10, 95))
    o_rating = (
        "強勁看多" if o_score_int >= 82 else
        "正向偏多" if o_score_int >= 66 else
        "中性觀望" if o_score_int >= 48 else
        "動能轉弱" if o_score_int >= 35 else "保守偏空"
    )
    outlook = ForwardOutlookReport(
        outlook_score=o_score_int, risk_equivalent=float(100 - o_score_int), rating=o_rating,
        trailing_pe=trailing_pe, forward_pe=forward_pe, implied_eps_growth_pct=implied_eps_g,
        target_mean_price=target_mean, target_upside_pct=target_upside, analyst_count=analyst_cnt,
        rev_accel_pct=rev_accel,
        strengths=o_strengths if o_strengths else ["各項前瞻動能指標目前維持中性水準"],
        warnings=o_warns,
    )
    if has_hist or has_outlook:
        st.session_state["fund_outlook_cache"][cache_key] = {
            "ts": time.time(),
            "quality": quality,
            "outlook": outlook,
            "yf_name": yf_name,
        }
    return quality, outlook, yf_name


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single_stock_price_and_chips(stock_id: str, fetch_start: str, end: str, fm_start: str, token: str):
    """快取個股歷史股價（自動嘗試 .TW 與 .TWO）、融資融券與三大法人買賣超"""
    clean_id = str(stock_id).strip().upper().replace(".TW", "").replace(".TWO", "")
    stk, resolved_sym = pd.DataFrame(), clean_id
    for suffix in [".TW", ".TWO"]:
        sym = f"{clean_id}{suffix}"
        df = _fetch_yf(sym, fetch_start, end)
        if not df.empty and len(df) > 10:
            stk, resolved_sym = df, sym
            break

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
    return stk, resolved_sym, clean_id, stk_chips, stk_status


# =========================================================
# v10.0 核心量化風控＋主動式雙軸運算引擎
# =========================================================
class InstitutionalQuantRiskEngine:
    """台股量化風控與主動進攻雙軸引擎（v10.0：動態 Beta、Z-Score、風險加速度、主動 Alpha、四象限）"""

    def __init__(self, config: Optional[RiskModelConfig] = None):
        self.cfg = config or RiskModelConfig()
        self.w = self.cfg.weights
        self.mri_weights = np.arange(1, self.cfg.window_days + 1, dtype=float)
        self.chip_weights = np.arange(1, self.cfg.chip_smooth_days + 1, dtype=float)

    def score_market_snapshot(self, row: pd.Series, news_risk_score: float = 50.0) -> float:
        """計算單日大盤系統風險快照分數（0~100）"""
        # 1) 美股科技與費城半導體動能
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

        # 2) 美債殖利率與新台幣匯率
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

        # 3) 台股大盤籌碼與位階
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

        # 4) 原油與黃金
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
        s_cmd = float(np.clip(s_cmd, 10, 100))

        # 5) 比特幣流動性風險偏好
        s_btc = 45.0
        btc_10d = row.get("btc_10d_pct", 0.0)
        if pd.notna(btc_10d):
            if btc_10d <= -12.0:
                s_btc += 35
            elif btc_10d <= -6.0:
                s_btc += 18
            elif btc_10d >= 8.0:
                s_btc -= 18
            elif btc_10d >= 4.0:
                s_btc -= 10
        s_btc = float(np.clip(s_btc, 10, 100))

        # 6) VIX 與新聞情緒
        s_vix = 42.0
        vix, vix_5d = row.get("vix_close", 15.0), row.get("vix_5d_pct", 0.0)
        if pd.notna(vix):
            if vix >= 26.0:
                s_vix += 42
            elif vix >= 20.0 or (vix >= 16.5 and pd.notna(vix_5d) and vix_5d >= 15.0):
                s_vix += 22
            elif vix <= 14.0:
                s_vix -= 15
        s_vix = float(np.clip(s_vix, 10, 100))
        s_sent = s_vix * 0.618 + news_risk_score * 0.382

        mri_snap = (
            s_us * self.w.w_us_tech + s_rf * self.w.w_rates_fx + s_tw * self.w.w_tw_market
            + s_cmd * self.w.w_commodities + s_btc * self.w.w_btc_liquidity + s_sent * self.w.w_vix_news
        )
        return float(np.clip(mri_snap, 0.0, 100.0))

    @staticmethod
    def score_stock_chips_snapshot(row: pd.Series) -> float:
        """計算單日個股籌碼風險快照分數（0~100，納入法人二階流向轉折）"""
        sm_5d, sm_10d = row.get("smart_5d_vol_ratio", np.nan), row.get("smart_10d_vol_ratio", np.nan)
        sm_prev5d = row.get("smart_prev5d_vol_ratio", np.nan)
        stk_m5, stk_m10 = row.get("margin_5d_chg_pct", np.nan), row.get("margin_10d_chg_pct", np.nan)

        if pd.isna(sm_5d) and pd.isna(stk_m5):
            cmf, dd_20 = row.get("stk_cmf_10", 0.0), row.get("stk_dd_20d", 0.0)
            score = 48.0
            if pd.notna(cmf):
                score += float(np.clip(-cmf * 75.0, -25.0, 30.0))
            if pd.notna(dd_20) and dd_20 <= -10.0:
                score += min(abs(dd_20 + 10.0) * 1.2, 16.0)
            return float(np.clip(score, 12.0, 92.0))

        score = 45.0
        m_util, sm_ratio = row.get("margin_util_pct", np.nan), row.get("short_margin_ratio", 0.0)
        trust_sell, trust_buy = row.get("trust_consec_sell", 0), row.get("trust_consec_buy", 0)
        trust_prev15 = row.get("trust_prev15_net", 0.0)

        if pd.notna(sm_5d):
            score += float(np.clip(-sm_5d * 2.1, -22.0, 26.0))
        # 二階法人轉折：由顯著賣超收斂轉為淨買超時，提早降低風險分
        if pd.notna(sm_5d) and pd.notna(sm_prev5d):
            if sm_prev5d <= -2.0 and sm_5d > 0.5:
                score -= 6.0
            elif sm_prev5d >= 2.0 and sm_5d < -1.5:
                score += 6.0

        if trust_sell >= 3 and trust_prev15 > 0:
            score += 14
        elif trust_sell >= 3:
            score += 8
        elif trust_buy >= 3:
            score -= 12
        if pd.notna(stk_m10) and pd.notna(sm_10d) and stk_m10 >= 4.0 and sm_10d <= -2.0:
            score += float(np.clip(stk_m10 * 1.2, 6.0, 16.0))
        if pd.notna(stk_m5) and stk_m5 <= -5.0:
            if pd.notna(sm_5d) and sm_5d < -1.0:
                score += 12
            elif pd.notna(sm_5d) and sm_5d >= 2.5:
                score -= 14
        if pd.notna(m_util) and m_util >= 45.0 and pd.notna(sm_5d) and sm_5d < 0:
            score += 8
        if pd.notna(sm_ratio) and sm_ratio >= 18.0:
            score -= 8
        return float(np.clip(score, 8.0, 98.0))

    @staticmethod
    def score_stock_trend_risk(row: pd.Series) -> float:
        """
        計算自適應技術趨勢風險分（以個股自身 ATR 正規化乖離與回撤）：
        讓低波動金融股與高波動電子股各自用自身 ATR 尺度衡量破線嚴重度。
        """
        close_p = float(row.get("known_stk_close", 0.0))
        vwap_20 = float(row.get("stk_vwap_20", close_p))
        atr = float(row.get("stk_atr_14", close_p * 0.025))
        atr_dd = _safe_float(row.get("stk_atr_norm_dd"))

        if pd.isna(atr) or atr <= 0:
            atr = max(close_p * 0.025, 0.01)

        t_risk = 45.0
        # 以 ATR 倍數計算現價與 20 日均價距離
        vwap_atr_diff = (close_p - vwap_20) / atr
        if vwap_atr_diff < 0:
            t_risk += min(abs(vwap_atr_diff) * 9.5, 30.0)
        else:
            t_risk -= min(vwap_atr_diff * 5.0, 12.0)

        # 以 ATR 正規化之 20 日高點回撤（超過 3.0 倍 ATR 視為顯著轉弱）
        if atr_dd is not None and atr_dd >= 3.0:
            t_risk += min((atr_dd - 3.0) * 6.0, 20.0)

        return float(np.clip(t_risk, 10.0, 95.0))

    def _get_lock_floor(self, row: pd.Series) -> float:
        """檢查外資賣壓/年線、美債/VIX 與大盤波動/回撤三項系統性條件，決定風險鎖定地板"""
        lock_cnt = 0
        f_5d, f_10d = row.get("mkt_foreign_5d_b", np.nan), row.get("mkt_foreign_10d_b", np.nan)
        if pd.notna(f_5d):
            if f_5d <= -180 or (f_5d < 0 and pd.notna(f_10d) and f_10d <= -450):
                lock_cnt += 1
        elif row.get("twii_below_ma240", 0) == 1:
            lock_cnt += 1
        tnx, vix = row.get("tnx_close", np.nan), row.get("vix_close", np.nan)
        if (pd.notna(tnx) and tnx >= 5.00) or (pd.notna(vix) and vix >= 22.0):
            lock_cnt += 1
        hv, dd = row.get("twii_hv_20", 0.0), row.get("twii_dd_60d", 0.0)
        if (pd.notna(hv) and hv >= 24.0) or (pd.notna(dd) and dd <= -10.0 and (pd.isna(f_5d) or f_5d < 100)):
            lock_cnt += 1
        return self.cfg.lock_floor_level2 if lock_cnt >= 3 else (self.cfg.lock_floor_level1 if lock_cnt >= 2 else 0.0)

    @staticmethod
    def _calc_prices(sri_score: int, mri_score: int, combined_score: int, row: pd.Series) -> Tuple[float, float]:
        """依 SRI 風險等級、ATR 波動幅度與基本面/展望綜合分數計算建議賣出價與下檔防守價"""
        close_p = float(row["known_stk_close"])
        high_15 = float(row.get("stk_high_15d", close_p))
        atr = float(row.get("stk_atr_14", close_p * 0.025))
        vwap_20 = float(row.get("stk_vwap_20", close_p))
        if pd.isna(atr) or atr <= 0:
            atr = close_p * 0.025
        eff_atr = min(atr, close_p * 0.065)
        if pd.isna(vwap_20) or vwap_20 <= 0:
            vwap_20 = close_p
        if pd.isna(high_15) or high_15 < close_p:
            high_15 = close_p
        q_bonus = 0.70 if combined_score >= 72 else (0.25 if combined_score >= 55 else -0.25)
        if sri_score >= 65:
            raw_sell = np.clip(close_p + (1.20 + max(0.0, q_bonus * 0.4)) * eff_atr, close_p * 1.040, close_p * 1.100)
            raw_stop = np.clip(close_p - 1.15 * eff_atr, close_p * 0.930, close_p * 0.970)
        elif sri_score >= 48:
            raw_sell = close_p + (1.85 + max(0.0, q_bonus * 0.7)) * eff_atr
            if close_p < vwap_20 <= close_p * 1.14 and mri_score >= 65:
                raw_sell = max(raw_sell, vwap_20 * 1.01)
            raw_sell = np.clip(raw_sell, close_p * 1.070, close_p * 1.160)
            raw_stop = np.clip(min(high_15 - 1.90 * eff_atr, close_p - 1.45 * eff_atr), close_p * 0.905, close_p * 0.960)
        else:
            raw_sell = np.clip(close_p + (2.80 + q_bonus) * eff_atr, close_p * 1.120, close_p * 1.250)
            raw_stop = np.clip(min(high_15 - 2.65 * eff_atr, close_p - 1.90 * eff_atr), close_p * 0.875, close_p * 0.945)
        sell_p = round_tw_tick(float(raw_sell))
        stop_p = round_tw_tick(float(raw_stop))
        if sell_p <= close_p:
            sell_p = round_tw_tick(close_p * 1.03)
        if stop_p >= close_p:
            stop_p = round_tw_tick(close_p * 0.97)
        return sell_p, stop_p

    def run_analysis(self, stock_id: str, lookback_days: int = 75, fast_batch: bool = False) -> Dict:
        """執行完整單股或批量雙軸分析，回傳風險指標、主動 Alpha、四象限、價位與歷史序列"""
        today = datetime.date.today()
        fetch_start = (today - datetime.timedelta(days=450)).strftime("%Y-%m-%d")
        fm_start = (today - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        end_str = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        token = get_finmind_token()

        twii, macro_data, mkt_chips, mkt_status = fetch_shared_market_data(fetch_start, end_str, fm_start, token)
        stk, resolved_sym, clean_id, stk_chips, stk_status = fetch_single_stock_price_and_chips(
            stock_id, fetch_start, end_str, fm_start, token
        )
        if stk.empty or twii.empty:
            raise ValueError(f"查無股票代號「{stock_id}」，請確認是否為有效的台股上市／上櫃代碼。")

        stk_close_clean = stk["Close"].dropna()
        latest_close_val = float(stk_close_clean.iloc[-1])
        latest_dd_20d = float((stk_close_clean.iloc[-1] / stk_close_clean.tail(20).max() - 1.0) * 100.0)

        # 查詢歷史財務體質與未來展望（含深跌目標價折價防呆）
        quality, outlook, yf_name = fetch_fundamentals_and_outlook_smart(
            clean_id, resolved_sym, latest_close_val, latest_dd_20d, token
        )
        name_map = fetch_tw_stock_name_map(token)
        stock_name = name_map.get(clean_id) or yf_name
        display_name = f"{clean_id}-{stock_name}" if stock_name else clean_id

        live_news = fetch_news_risk_score("") if fast_batch else fetch_news_risk_score(stock_id)
        mkt_note = "真實籌碼" if not mkt_chips.empty else f"備援({mkt_status})"
        stk_note = "真實籌碼" if not stk_chips.empty else f"資金流代理({stk_status})"

        # 對齊大盤與個股日線序列並計算動態與主動技術指標
        df = pd.DataFrame(index=twii.index)
        twii_c = twii["Close"]
        stk_h = stk["High"].reindex(df.index).ffill()
        stk_l = stk["Low"].reindex(df.index).ffill()
        stk_c = stk["Close"].reindex(df.index).ffill()
        stk_v = stk["Volume"].reindex(df.index).ffill()

        twii_ma60 = twii_c.rolling(60).mean()
        twii_ma240 = twii_c.rolling(240, min_periods=120).mean()
        df["twii_bias_60"] = (twii_c / twii_ma60 - 1) * 100
        df["twii_below_ma60"] = (twii_c < twii_ma60).astype(int)
        df["twii_below_ma240"] = (twii_c < twii_ma240).astype(int)
        df["twii_hv_20"] = twii_c.pct_change().rolling(20).std() * np.sqrt(252) * 100
        df["twii_dd_60d"] = (twii_c / twii_c.rolling(60, min_periods=10).max() - 1) * 100

        prev_c = stk_c.shift(1)
        tr = pd.concat([(stk_h - stk_l), (stk_h - prev_c).abs(), (stk_l - prev_c).abs()], axis=1).max(axis=1)
        df["stk_atr_14"] = tr.rolling(14).mean()
        df["stk_atr_10"] = tr.rolling(10).mean()
        df["stk_atr_60"] = tr.rolling(60, min_periods=20).mean()
        df["vol_squeeze_ratio"] = df["stk_atr_10"] / df["stk_atr_60"].replace(0, np.nan)
        df["min_sq_5d"] = df["vol_squeeze_ratio"].rolling(5, min_periods=1).min()

        df["stk_high_15d"] = stk_h.rolling(15).max()
        df["stk_high_20d"] = stk_h.rolling(20, min_periods=5).max()
        df["stk_atr_norm_dd"] = (df["stk_high_20d"] - stk_c) / df["stk_atr_14"].replace(0, np.nan)
        df["stk_vwap_20"] = (stk_c * stk_v).rolling(20).sum() / stk_v.rolling(20).sum().replace(0, np.nan)
        df["stk_vol_ratio_20"] = stk_v / stk_v.rolling(20, min_periods=5).mean().replace(0, np.nan)
        df["stk_ret_10d"] = stk_c.pct_change(10) * 100.0

        # 相對大盤強弱指標 (RS Momentum)
        rs_line = stk_c / twii_c.replace(0, np.nan)
        rs_ma20 = rs_line.rolling(20).mean()
        rs_high60 = rs_line.rolling(60, min_periods=20).max()
        df["rs_vs_ma20_pct"] = (rs_line / rs_ma20 - 1.0) * 100.0
        df["rs_new_high_60d"] = (rs_line >= rs_high60 * 0.995).astype(int)

        hl_range = (stk_h - stk_l).replace(0, np.nan)
        mfm = (((stk_c - stk_l) - (stk_h - stk_c)) / hl_range).fillna(0.0)
        df["stk_cmf_10"] = (mfm * stk_v).rolling(10).sum() / stk_v.rolling(10).sum().replace(0, np.nan)
        df["stk_dd_20d"] = (stk_c / stk_c.rolling(20, min_periods=5).max() - 1) * 100

        def align(k: str) -> pd.Series:
            """將各總經指標對齊至大盤交易日曆並以昨日收盤前向填補"""
            return pd.Series(np.nan, index=df.index) if macro_data[k].empty else macro_data[k]["Close"].reindex(df.index, method="ffill")

        sox, ndx, tnx = align("SOX"), align("NDX"), align("TNX")
        df["sox_5d_pct"], df["sox_10d_pct"] = sox.pct_change(5) * 100, sox.pct_change(10) * 100
        df["ndx_5d_pct"], df["ndx_10d_pct"] = ndx.pct_change(5) * 100, ndx.pct_change(10) * 100
        df["tnx_close"], df["tnx_20d_chg"] = tnx, tnx - tnx.shift(20)
        df["usdtwd_10d_pct"] = align("USDTWD").pct_change(10) * 100
        df["gold_10d_pct"] = align("GOLD").pct_change(10) * 100
        oil = align("OIL")
        df["oil_close"], df["oil_10d_pct"] = oil, oil.pct_change(10) * 100
        df["btc_10d_pct"] = align("BTC").pct_change(10) * 100
        vix = align("VIX")
        df["vix_close"], df["vix_5d_pct"] = vix, vix.pct_change(5) * 100

        if not mkt_chips.empty:
            mc = mkt_chips.reindex(df.index).ffill()
            if "mkt_margin_bal" in mc:
                df["mkt_margin_rank_60d"] = mc["mkt_margin_bal"].rolling(60, min_periods=20).rank(pct=True)
                df["mkt_margin_5d_pct"] = mc["mkt_margin_bal"].pct_change(5) * 100
            if "mkt_foreign_net_b" in mc:
                df["mkt_foreign_5d_b"] = mc["mkt_foreign_net_b"].rolling(5).sum()
                df["mkt_foreign_10d_b"] = mc["mkt_foreign_net_b"].rolling(10).sum()

        if not stk_chips.empty:
            sc = stk_chips.reindex(df.index).ffill()
            if "margin_bal" in sc:
                df["margin_5d_chg_pct"] = sc["margin_bal"].pct_change(5) * 100
                df["margin_10d_chg_pct"] = sc["margin_bal"].pct_change(10) * 100
                df["margin_util_pct"] = sc.get("margin_util_pct", np.nan)
                df["short_margin_ratio"] = sc.get("short_margin_ratio", 0.0)
            if "smart_money_net" in sc:
                v5, v10 = stk_v.rolling(5).sum().replace(0, np.nan), stk_v.rolling(10).sum().replace(0, np.nan)
                df["smart_5d_vol_ratio"] = (sc["smart_money_net"].rolling(5).sum() / v5) * 100
                df["smart_prev5d_vol_ratio"] = df["smart_5d_vol_ratio"].shift(5)
                df["smart_10d_vol_ratio"] = (sc["smart_money_net"].rolling(10).sum() / v10) * 100
                is_sell, is_buy = (sc["trust_net"] < 0).astype(int), (sc["trust_net"] > 0).astype(int)
                df["trust_consec_sell"] = np.where(is_sell == 1, is_sell.groupby((is_sell != is_sell.shift()).cumsum()).cumsum(), 0)
                df["trust_consec_buy"] = np.where(is_buy == 1, is_buy.groupby((is_buy != is_buy.shift()).cumsum()).cumsum(), 0)
                df["trust_prev15_net"] = sc["trust_net"].shift(3).rolling(15).sum()

        # 計算個股 60 日 Rolling Beta、自適應籌碼痛點與逆風韌性
        rolling_beta = compute_rolling_beta(stk_c, twii_c, window=60, min_periods=20)
        margin_s = stk_chips["margin_bal"] if (not stk_chips.empty and "margin_bal" in stk_chips) else None
        inst_s = stk_chips["smart_money_net"] if (not stk_chips.empty and "smart_money_net" in stk_chips) else None
        pain_report = compute_chip_pain_index(stk, margin_s)
        downside_report = compute_downside_alpha(stk, twii, inst_s, lookback_days=60)
        resilience_bonus = -5.0 if downside_report.has_badge else 0.0

        work_df = df.dropna(subset=["stk_atr_14"]).tail(lookback_days + 15).copy()
        work_df["known_stk_close"] = stk_c.reindex(work_df.index)
        n = len(work_df)

        mri_snaps = np.array([
            self.score_market_snapshot(work_df.iloc[i], news_risk_score=(live_news if i == n - 1 else 50.0))
            for i in range(n)
        ], dtype=float)
        chip_snaps = np.array([self.score_stock_chips_snapshot(work_df.iloc[i]) for i in range(n)], dtype=float)

        combined_q = int(round(quality.quality_score * 0.5 + outlook.outlook_score * 0.5))
        history = []
        base_sri_seq: List[float] = []
        latest_overlay = None
        w_m, w_c = self.cfg.window_days, self.cfg.chip_smooth_days

        for i in range(max(0, n - lookback_days), n):
            is_last = (i == n - 1)
            m_win = mri_snaps[max(0, i - w_m + 1) : i + 1]
            c_win = chip_snaps[max(0, i - w_c + 1) : i + 1]
            row = work_df.iloc[i]

            # 大盤 MRI
            wma_10 = float(np.average(m_win, weights=self.mri_weights[-len(m_win):]))
            mom = float(np.mean(m_win[-3:]) - np.mean(m_win[:-3])) if len(m_win) >= 6 else 0.0
            trend_adj = min(mom * 0.45, 13.0) if mom >= 5.0 else (max(mom * 0.20, -6.0) if mom <= -7.0 else 0.0)
            high_days = int(np.sum(m_win >= 65.0))
            persist_adj = 7.0 if high_days >= 7 else (3.5 if high_days >= 5 else 0.0)
            mri_score = int(np.clip(round(max(wma_10 + trend_adj + persist_adj, self._get_lock_floor(row))), 0, 100))

            # 動態權重狀態機 × Rolling Beta
            w_mri_dyn, w_idio_dyn, base_w_mri, regime_name = get_dynamic_regime_weights(mri_score, rolling_beta)

            # 自適應籌碼痛點修正
            chip_smooth = float(np.average(c_win, weights=self.chip_weights[-len(c_win):]))
            adj_chip = float(np.clip(chip_smooth + pain_report.risk_adj, 5.0, 98.0))
            idio = (
                adj_chip * self.w.w_stock_chip_in_idio
                + quality.risk_equivalent * self.w.w_stock_health_in_idio
                + outlook.risk_equivalent * self.w.w_stock_outlook_in_idio
            )

            # 訊號背離非線性懲罰（趨勢風險改採 ATR 自適應版）
            trend_risk = self.score_stock_trend_risk(row)
            div_penalty, div_warn = detect_divergence_penalty(
                quality_risk=quality.risk_equivalent,
                outlook_risk=outlook.risk_equivalent,
                chip_risk=adj_chip,
                trend_risk=trend_risk,
            )

            base_sri = mri_score * w_mri_dyn + idio * w_idio_dyn + div_penalty + resilience_bonus
            if adj_chip > 68.0:
                base_sri += min((adj_chip - 68.0) * 0.45, 10.0)
            base_sri_seq.append(base_sri)

            # 風險加速度（二階斜率）微調
            vel_3d, vel_tag, vel_adj = evaluate_sri_velocity(base_sri_seq)
            sri_score = int(np.clip(round(base_sri + vel_adj), 0, 100))

            if is_last:
                atr_dd_val = _safe_float(row.get("stk_atr_norm_dd"))
                dyn_note = (
                    f"60日 Beta={rolling_beta:.2f}（大盤權重由基準 {int(base_w_mri*100)}% 動態調整為 {int(round(w_mri_dyn*100))}%）；"
                    f"近3日風險斜率 {vel_3d:+.1f} 分（{vel_tag}）"
                )
                dyn_report = DynamicFactorReport(
                    rolling_beta_60d=rolling_beta,
                    base_w_mri=base_w_mri,
                    beta_adj_w_mri=w_mri_dyn,
                    sri_velocity_3d=vel_3d,
                    velocity_tag=vel_tag,
                    velocity_adj=vel_adj,
                    atr_norm_dd=round(atr_dd_val, 2) if atr_dd_val is not None else None,
                    note=dyn_note,
                )
                active_alpha = compute_active_alpha_and_quadrant(
                    work_df, sri_score, outlook.outlook_score, downside_report
                )
                latest_overlay = AdvancedRiskOverlay(
                    regime_name=regime_name,
                    w_mri_dynamic=w_mri_dyn,
                    w_idio_dynamic=w_idio_dyn,
                    divergence_penalty=div_penalty,
                    divergence_warning=div_warn,
                    pain_report=pain_report,
                    downside_alpha=downside_report,
                    dynamic_factors=dyn_report,
                    active_alpha=active_alpha,
                )

            sell_p, stop_p = self._calc_prices(sri_score, mri_score, combined_q, row)
            history.append({
                "date": work_df.index[i],
                "close": round(float(row["known_stk_close"]), 2),
                "vwap_20": round(float(row["stk_vwap_20"]), 2) if pd.notna(row.get("stk_vwap_20")) else np.nan,
                "mri": mri_score,
                "sri": sri_score,
                "chip": int(round(adj_chip)),
                "sell_p": sell_p,
                "stop_p": stop_p,
            })

        hist_df = pd.DataFrame(history).set_index("date")
        latest = history[-1]
        return {
            "symbol": resolved_sym,
            "display_name": display_name,
            "date": latest["date"].strftime("%Y-%m-%d"),
            "close": latest["close"],
            "mri": latest["mri"],
            "sri": latest["sri"],
            "chip": latest["chip"],
            "sell_p": latest["sell_p"],
            "stop_p": latest["stop_p"],
            "quality": quality,
            "outlook": outlook,
            "overlay": latest_overlay,
            "mkt_note": mkt_note,
            "stk_note": stk_note,
            "hist_df": hist_df,
        }


# =========================================================
# 卡片渲染：主動式 Alpha 四象限、動態風控偵測、體質、展望、圖表
# =========================================================
def _metric_tag(val, good_cond, bad_cond, good_txt, mid_txt, bad_txt, empty_txt="無資料"):
    """依數值條件回傳對應顏色與文字的指標狀態標籤"""
    if val is None or pd.isna(val):
        return f'<span class="m-tag" style="color:#6e7681;">{empty_txt}</span>'
    if good_cond(val):
        return f'<span class="m-tag tag-good">{good_txt}</span>'
    if bad_cond(val):
        return f'<span class="m-tag tag-bad">{bad_txt}</span>'
    return f'<span class="m-tag tag-mid">{mid_txt}</span>'


def render_active_alpha_card(aa: ActiveAlphaReport):
    """渲染主動式進攻評分與四象限決策卡片"""
    score_col = "#3fb950" if aa.alpha_score >= 60 else ("#ff7b72" if aa.alpha_score < 42 else "#e3b341")
    rs_t = (
        '<span class="m-tag tag-good">創60日新高</span>' if aa.rs_new_high_60d else
        _metric_tag(aa.rs_vs_ma20_pct, lambda x: x >= 2.0, lambda x: x <= -3.0, "強於大盤", "同步大盤", "弱於大盤")
    )
    sq_t = (
        '<span class="m-tag tag-good">壓縮突破🔥</span>' if aa.is_squeeze_breakout else
        _metric_tag(aa.vol_squeeze_ratio, lambda x: x <= 0.80, lambda x: x >= 1.35, "壓縮蓄勢", "正常波動", "劇烈放大")
    )
    acc_t = (
        '<span class="m-tag tag-good">低檔吸納💎</span>' if aa.is_stealth_accum else
        _metric_tag(aa.stealth_accum_pct, lambda x: x >= 2.5, lambda x: x <= -2.5, "法人偏買", "籌碼平穩", "法人調節", "代理監控")
    )

    sq_str = f"{aa.vol_squeeze_ratio:.2f}x" if aa.vol_squeeze_ratio is not None else "N/A"
    sig_html = "".join([f"<div>⚡ {s}</div>" for s in aa.signals])

    _render_html(f"""<div class="risk-card">
        <div class="card-title">🎯 主動式進攻因子 (Active Alpha) ＆ 四象限定位</div>
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:4px;">
            <span style="font-size:1.45rem;font-weight:700;color:{score_col};">{aa.alpha_score} <span style="font-size:0.95rem;color:#8b949e;">/ 100</span> 【{aa.rating}】</span>
            <span class="{aa.quadrant_badge_cls}">{aa.quadrant}</span>
        </div>
        <div class="card-sub" style="margin-top:4px;">{aa.quadrant_desc}</div>
        <div class="outlook-grid">
            <div class="metric-box">
                <div class="m-label">相對大盤強弱 (RS)</div>
                <div class="m-val">{_fmt_num(aa.rs_vs_ma20_pct, '%', signed=True)}</div>
                <div class="m-note">距 20 日 RS 均線</div>
                {rs_t}
            </div>
            <div class="metric-box">
                <div class="m-label">波動率壓縮比</div>
                <div class="m-val">{sq_str}</div>
                <div class="m-note">10日 / 60日 ATR</div>
                {sq_t}
            </div>
            <div class="metric-box">
                <div class="m-label">法人 5 日吸納</div>
                <div class="m-val">{_fmt_num(aa.stealth_accum_pct, '%', signed=True)}</div>
                <div class="m-note">土洋買超佔量比</div>
                {acc_t}
            </div>
        </div>
        <div class="health-list">{sig_html}</div>
        </div>""")


def render_advanced_overlay_card(ov: AdvancedRiskOverlay):
    """渲染動態量化風控偵測卡片（Beta 狀態機、風險加速度、自適應籌碼痛點、逆風韌性、背離警示）"""
    dyn = ov.dynamic_factors
    vel_cls = "badge-red" if dyn.velocity_tag == "加速惡化" else ("badge-yellow" if dyn.velocity_tag == "緩步升溫" else ("badge-green" if dyn.velocity_tag == "風險收斂" else "badge-blue"))

    badges_html = f'<span class="badge-yellow">狀態機：{ov.regime_name}</span>'
    badges_html += f'<span class="{vel_cls}">斜率：{dyn.velocity_tag} ({dyn.sri_velocity_3d:+.1f})</span>'
    if ov.downside_alpha.has_badge:
        badges_html += '<span class="badge-green">🛡️ 逆風抗跌徽章</span>'
    if ov.pain_report.status == "多殺多預警":
        badges_html += '<span class="badge-red">🔥 多殺多預警</span>'
    elif ov.pain_report.status == "籌碼乾淨":
        badges_html += '<span class="badge-green">💎 籌碼乾淨</span>'

    div_html = ""
    if ov.divergence_warning:
        div_html = (
            f'<div style="background:rgba(255,123,114,0.16); border:1px solid #ff7b72; '
            f'border-radius:8px; padding:10px 12px; margin-top:8px; margin-bottom:6px; '
            f'color:#ff7b72; font-weight:600; font-size:0.85rem;">'
            f'{ov.divergence_warning}（風險分 +{ov.divergence_penalty:.0f}）</div>'
        )

    _render_html(f"""<div class="risk-card">
        <div class="card-title">🧠 動態風控引擎偵測（Beta 狀態機 / 風險加速度 / Z-Score 痛點 / 逆風韌性）</div>
        <div style="margin:6px 0 8px 0;">{badges_html}</div>
        {div_html}
        <div class="health-list">
        <div>• <b>動態調校：</b>{dyn.note}</div>
        <div>• <b>籌碼痛點 (20日VWAP)：</b>{ov.pain_report.note}</div>
        <div>• <b>逆風韌性 (Downside Alpha)：</b>{ov.downside_alpha.note}</div>
        </div>
        </div>""")


def render_health_card(q: FundamentalQualityReport):
    """渲染歷史財務體質健檢卡片"""
    roe_t = _metric_tag(q.roe_pct, lambda x: x >= 15, lambda x: x < 5, "優異", "穩健", "偏弱")
    op_t = _metric_tag(q.op_margin_pct, lambda x: x >= 15, lambda x: x < 3, "高毛利", "普通", "偏薄")
    rev_t = _metric_tag(q.rev_growth_pct, lambda x: x >= 15, lambda x: x <= -10, "高成長", "平穩", "衰退")
    pe_t = _metric_tag(q.pe, lambda x: 0 < x <= 16, lambda x: x >= 38 or x < 0, "便宜", "合理", "偏貴/虧損")
    pb_t = _metric_tag(q.pb, lambda x: 0 < x <= 2.0, lambda x: x >= 6.0, "低估", "合理", "溢價")
    pb_str = f"{q.pb:.2f}x" if q.pb is not None else "N/A"
    str_html = "".join([f"<div>✅ {s}</div>" for s in q.strengths])
    warn_html = "".join([f"<div>⚠️ {w}</div>" for w in q.warnings]) if q.warnings else "<div>✨ 目前無明顯財務警訊</div>"
    _render_html(f"""<div class="risk-card">
        <div class="card-title">📊 個股財務體質健檢（來源：{q.source_note}）</div>
        <div style="font-size:1.35rem;font-weight:700;color:#f0f6fc;">{q.quality_score} <span style="font-size:0.95rem;color:#8b949e;">/ 100</span> &nbsp;【{q.rating}】</div>
        <div class="metric-grid">
        <div class="metric-box"><div class="m-label">ROE 報酬率</div><div class="m-val">{_fmt_num(q.roe_pct, '%')}</div>{roe_t}</div>
        <div class="metric-box"><div class="m-label">營業利益率</div><div class="m-val">{_fmt_num(q.op_margin_pct, '%')}</div>{op_t}</div>
        <div class="metric-box"><div class="m-label">營收年增率</div><div class="m-val">{_fmt_num(q.rev_growth_pct, '%', signed=True)}</div>{rev_t}</div>
        <div class="metric-box"><div class="m-label">本益比 PE</div><div class="m-val">{_fmt_num(q.pe, 'x')}</div>{pe_t}</div>
        <div class="metric-box"><div class="m-label">淨值比 PB</div><div class="m-val">{pb_str}</div>{pb_t}</div>
        </div>
        <div class="health-list">{str_html}{warn_html}</div>
        </div>""")


def render_outlook_card(o: ForwardOutlookReport):
    """渲染未來展望與法人預期卡片"""
    eps_t = _metric_tag(o.implied_eps_growth_pct, lambda x: x >= 15, lambda x: x <= -5, "看增", "平穩", "看減", "無預估")
    tgt_t = _metric_tag(o.target_upside_pct, lambda x: x >= 18, lambda x: x <= -5, "具空間", "合理", "偏貴", "無外資追蹤")
    acc_t = _metric_tag(o.rev_accel_pct, lambda x: x >= 10, lambda x: x <= -5, "加速中", "持平", "鈍化", "無資料")
    pe_sub = f"預估 {o.forward_pe}x / 過去 {o.trailing_pe}x" if (o.forward_pe and o.trailing_pe) else "需雙 PE 數據"
    tgt_sub = f"目標 {o.target_mean_price:.1f} 元" if o.target_mean_price else "中小型股常見"
    acc_sub = "近 3 月 vs 近 12 月"
    score_col = "#3fb950" if o.outlook_score >= 66 else ("#ff7b72" if o.outlook_score < 48 else "#e3b341")
    str_html = "".join([f"<div>🚀 {s}</div>" for s in o.strengths])
    warn_html = "".join([f"<div>⚠️ {w}</div>" for w in o.warnings]) if o.warnings else ""
    _render_html(f"""<div class="risk-card">
        <div class="card-title">🔮 未來展望與法人預期</div>
        <div style="font-size:1.35rem;font-weight:700;color:{score_col};">{o.outlook_score} <span style="font-size:0.95rem;color:#8b949e;">/ 100</span> &nbsp;【{o.rating}】</div>
        <div class="outlook-grid">
        <div class="metric-box">
        <div class="m-label">預估獲利成長</div>
        <div class="m-val">{_fmt_num(o.implied_eps_growth_pct, '%', signed=True)}</div>
        <div class="m-note">{pe_sub}</div>
        {eps_t}
        </div>
        <div class="metric-box">
        <div class="m-label">法人目標空間</div>
        <div class="m-val">{_fmt_num(o.target_upside_pct, '%', signed=True)}</div>
        <div class="m-note">{tgt_sub}</div>
        {tgt_t}
        </div>
        <div class="metric-box">
        <div class="m-label">營收動能加速</div>
        <div class="m-val">{_fmt_num(o.rev_accel_pct, '%', signed=True)}</div>
        <div class="m-note">{acc_sub}</div>
        {acc_t}
        </div>
        </div>
        <div class="health-list">{str_html}{warn_html}</div>
        </div>""")


def render_dark_chart(res: Dict):
    """繪製雙層深色互動圖表：上層為收盤價、20日VWAP與動態賣出/防守價；下層為大盤 MRI 與個股 SRI"""
    df = res["hist_df"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.62, 0.38]
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="收盤價", line=dict(color="#58a6ff", width=2.5)), row=1, col=1)
    if "vwap_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_20"], name="20日均價(VWAP)", line=dict(color="#8b949e", width=1.2, dash="dashdot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sell_p"], name="建議賣出價", line=dict(color="#d2a8ff", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["stop_p"], name="下檔防守價", line=dict(color="#f0883e", width=1.6, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["mri"], name="大盤風險 (MRI)", line=dict(color="#ff7b72", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sri"], name="個股風險 (SRI)", line=dict(color="#3fb950", width=2, dash="dash")), row=2, col=1)

    fig.add_hrect(y0=65, y1=100, fillcolor="rgba(255,123,114,0.12)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=48, fillcolor="rgba(63,185,80,0.10)", line_width=0, row=2, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color="#ff7b72", opacity=0.7, row=2, col=1)
    fig.add_hline(y=48, line_dash="dot", line_color="#3fb950", opacity=0.7, row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#161b22",
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="股價", gridcolor="#30363d", row=1, col=1)
    fig.update_yaxes(title_text="風險分", range=[0, 102], gridcolor="#30363d", row=2, col=1)
    fig.update_xaxes(gridcolor="#30363d")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# 側邊欄：FinMind Token 設定與 v10 模型說明
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 系統連線設定")
    st.text_input(
        "FinMind API Token（選填）",
        type="password",
        key="user_finmind_token",
        help="若遇到免費限流，可至 finmindtrade.com 免費申請 Token 填入。",
    )
    st.caption(
        "風控燈號 (SRI / MRI)：\n🟢 安全 (<48)\n🟡 中等 (48~64)\n🔴 危險 (≥65)\n\n"
        "v10 動態狀態機 × Rolling Beta：\n"
        "• MRI < 45：基準 15% × Beta\n"
        "• 45 ≤ MRI < 70：基準 40% × Beta\n"
        "• MRI ≥ 70：基準 65% × Beta\n\n"
        "四象限定位（SRI 52 / Alpha 60）：\n"
        "• 🚀 主升段首選：低險＋高攻\n"
        "• 🛡️ 防禦盤整：低險＋低攻\n"
        "• ⚡ 高檔博弈：高險＋高攻\n"
        "• 🚫 弱勢迴避：高險＋低攻"
    )


# =========================================================
# 主介面：雙分頁（單股雙軸診斷 / 多股批量掃描）
# =========================================================
st.markdown("## 🛡️ 台股量化風控儀表板 v10.1")
tab1, tab2 = st.tabs(["🔍 單股雙軸診斷", "📋 多股批量掃描"])
engine = InstitutionalQuantRiskEngine()

# 分頁 1：單股雙軸診斷
with tab1:
    col_in, col_btn = st.columns([3, 1.2])
    with col_in:
        stock_input = st.text_input(
            "輸入台股代號（上市／上櫃自動偵測）",
            value="2330",
            placeholder="例如 2330、2382、6274",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("一鍵分析", type="primary", use_container_width=True)

    if stock_input.strip():
        with st.spinner(f"正在運算 {stock_input.strip()} 動態風控、主動 Alpha 與前瞻模型..."):
            try:
                res = engine.run_analysis(stock_input.strip())
                m_icon, m_stat, m_col, m_cls = get_traffic_light(res["mri"])
                s_icon, s_stat, s_col, s_cls = get_traffic_light(res["sri"])
                sell_pct = (res["sell_p"] / res["close"] - 1) * 100
                stop_pct = (res["stop_p"] / res["close"] - 1) * 100
                ov: AdvancedRiskOverlay = res["overlay"]
                dyn = ov.dynamic_factors
                aa = ov.active_alpha
                mri_weight_pct = int(round(ov.w_mri_dynamic * 100))

                _render_html(f"### {res['display_name']} &nbsp; <span class='{aa.quadrant_badge_cls}' style='vertical-align:middle;'>{aa.quadrant}</span>")
                st.caption(f"基準日：{res['date']} ｜ 收盤價：{res['close']} 元 ｜ 60日 Beta：{dyn.rolling_beta_60d:.2f} ｜ 籌碼：{res['stk_note']}")

                # 第一排：個股風險 SRI 與大盤風險 MRI（含動態 Beta 權重與風險斜率）
                c1, c2 = st.columns(2)
                with c1:
                    _render_html(f"""<div class="risk-card">
                        <div class="card-title">個股風險指數 (SRI)</div>
                        <div class="card-value" style="color:{s_col};">{s_icon} {res['sri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
                        <div class="card-sub"><span class="{s_cls}">{s_stat}</span> 3日斜率 {dyn.sri_velocity_3d:+.1f}｜籌碼 {res['chip']}</div>
                        </div>""")
                with c2:
                    _render_html(f"""<div class="risk-card">
                        <div class="card-title">大盤風險指數 (MRI)</div>
                        <div class="card-value" style="color:{m_col};">{m_icon} {res['mri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
                        <div class="card-sub"><span class="{m_cls}">{m_stat}</span> Beta權重 {mri_weight_pct}%</div>
                        </div>""")

                # 第二排：動態建議賣出價與下檔防守價
                c3, c4 = st.columns(2)
                with c3:
                    _render_html(f"""<div class="risk-card">
                        <div class="card-title">建議賣出價（目標調節區）</div>
                        <div class="card-value" style="color:#d2a8ff;">{res['sell_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
                        <div class="card-sub">距收盤 {sell_pct:+.1f}%</div>
                        </div>""")
                with c4:
                    _render_html(f"""<div class="risk-card">
                        <div class="card-title">下檔防守價</div>
                        <div class="card-value" style="color:#f0883e;">{res['stop_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
                        <div class="card-sub">距收盤 {stop_pct:+.1f}%</div>
                        </div>""")

                # 第三排：主動式進攻因子與四象限卡片
                render_active_alpha_card(aa)

                # 第四排：動態量化風控偵測卡片
                render_advanced_overlay_card(ov)

                # 第五排：未來展望與法人預期卡片
                render_outlook_card(res["outlook"])

                # 第六排：歷史財務體質健檢卡片
                render_health_card(res["quality"])

                # 第七排：互動式走勢圖
                render_dark_chart(res)

            except Exception as e:
                st.error(f"分析失敗：{e}")

# 分頁 2：多股批量掃描（支援依四象限與主動 Alpha 篩選）
with tab2:
    batch_input = st.text_area(
        "輸入多檔代號（支援逗號、空白或換行分隔）",
        value="2330, 2454, 2382, 6274, 2881",
        height=72,
    )
    if st.button("開始批量掃描", type="primary", use_container_width=True):
        raw_tokens = re.split(r"[,，\s\n]+", batch_input.strip())
        tickers = list(dict.fromkeys([t.strip().upper() for t in raw_tokens if t.strip()]))
        if not tickers:
            st.warning("請先輸入至少一檔台股代號。")
        else:
            rows = []
            prog = st.progress(0)
            status_txt = st.empty()
            for idx, tid in enumerate(tickers):
                status_txt.caption(f"正在掃描 ({idx + 1}/{len(tickers)})：{tid} ...")
                try:
                    r = engine.run_analysis(tid, lookback_days=20, fast_batch=True)
                    m_i, _, _, _ = get_traffic_light(r["mri"])
                    s_i, s_st, _, _ = get_traffic_light(r["sri"])
                    sell_pct = (r["sell_p"] / r["close"] - 1) * 100
                    ov_b: AdvancedRiskOverlay = r["overlay"]
                    dyn_b = ov_b.dynamic_factors
                    aa_b = ov_b.active_alpha

                    tags = []
                    if aa_b.is_squeeze_breakout:
                        tags.append("🔥點火")
                    if aa_b.is_stealth_accum:
                        tags.append("🧲吸納")
                    if ov_b.downside_alpha.has_badge:
                        tags.append("🛡️抗跌")
                    if ov_b.divergence_warning:
                        tags.append("⚠️利多背離")
                    if ov_b.pain_report.status == "多殺多預警":
                        tags.append("💥多殺多")
                    elif ov_b.pain_report.status == "籌碼乾淨":
                        tags.append("💎乾淨")

                    rows.append({
                        "標的": r["display_name"],
                        "四象限": aa_b.quadrant,
                        "收盤價": r["close"],
                        "個股風險": f"{s_i} {r['sri']} ({dyn_b.sri_velocity_3d:+.0f})",
                        "主動Alpha": f"{aa_b.alpha_score} ({aa_b.rating})",
                        "Beta": dyn_b.rolling_beta_60d,
                        "進階訊號": " ".join(tags) if tags else "-",
                        "建議賣出價": f"{r['sell_p']:.1f} ({sell_pct:+.1f}%)",
                        "下檔防守價": r["stop_p"],
                        "大盤MRI": f"{m_i} {r['mri']}",
                    })
                except Exception as e:
                    rows.append({
                        "標的": tid,
                        "四象限": "-",
                        "收盤價": None,
                        "個股風險": f"失敗({str(e)[:12]})",
                        "主動Alpha": "-",
                        "Beta": None,
                        "進階訊號": "-",
                        "建議賣出價": "-",
                        "下檔防守價": None,
                        "大盤MRI": "-",
                    })
                prog.progress((idx + 1) / len(tickers))
            prog.empty()
            status_txt.empty()
            st.session_state["batch_results"] = pd.DataFrame(rows)

    if "batch_results" in st.session_state and not st.session_state["batch_results"].empty:
        st.dataframe(st.session_state["batch_results"], use_container_width=True, hide_index=True)

# 底部免責聲明
_render_html('<div class="disclaimer">本系統所有數據與風險指標皆由公開資訊與量化演算法自動運算產生，僅供風險控管參考，絕非投資買賣建議。<br>投資人應獨立判斷並自負盈虧與法律責任。</div>')

