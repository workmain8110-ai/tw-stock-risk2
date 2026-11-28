from __future__ import annotations

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
    page_title="台股量化風控儀表板 v10.2",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _render_html(html: str) -> None:
    """移除每行前導空白與空行，徹底避免 Streamlit Markdown 將 HTML 誤判為縮排程式碼區塊"""
    clean = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


# 自訂深色主題樣式：風險卡片、狀態徽章、四象限標籤、指標網格與免責聲明
_render_html(
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
    """
)

# 內建熱門台股中文名稱備援表（當 API 限流時仍可正確顯示中文）
FALLBACK_STOCK_NAMES: Dict[str, str] = {
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
        return str(st.session_state["user_finmind_token"]).strip()
    try:
        return str(st.secrets.get("FINMIND_TOKEN", "")).strip()
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


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """跨版本相容的百分比變化計算（不偷偷前向填補，避免停牌日產生假性 0% 漲跌）"""
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(periods)
    return s / prev.where(prev != 0.0) - 1.0


def _dedup_sort_index(df: pd.DataFrame) -> pd.DataFrame:
    """移除重複日期索引並按時間排序，避免 reindex 報錯"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


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


def rolling_zscore(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    clip_val: float = 4.0,
) -> pd.Series:
    """逐日滾動 Z-Score（僅使用過去 window 日自身分佈，無前視偏誤）"""
    s = pd.to_numeric(series, errors="coerce")
    roll = s.rolling(window=window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std(ddof=0)

    valid_obs = s.notna() & mean.notna() & std.notna()
    safe_std = std.where(std >= 1e-8)
    z = (s - mean) / safe_std
    z = z.where(~(valid_obs & (std < 1e-8)), 0.0)
    return z.clip(-clip_val, clip_val)


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
    w_us_tech: float = 0.200
    w_rates_fx: float = 0.180
    w_tw_market: float = 0.238
    w_commodities: float = 0.120
    w_btc_liquidity: float = 0.100
    w_vix_news: float = 0.162
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
    lock_floor_level1: float = 68.0
    lock_floor_level2: float = 78.0


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
    """自適應籌碼痛點指數報告：結合 20 日典型價 VWAP 乖離、20 日融資變化與 60 日自適應 Z-Score"""
    status: str
    risk_adj: float
    vwap_20: Optional[float] = None
    price_vs_vwap_pct: Optional[float] = None
    margin_chg_20d_pct: Optional[float] = None
    vwap_zscore_60d: Optional[float] = None
    margin_zscore_60d: Optional[float] = None
    note: str = ""


@dataclass
class DownsideAlphaReport:
    """逆風韌性係數報告：統計近 60 日大盤單日跌逾 1% 時的個股表現與法人買超"""
    has_badge: bool
    down_days_count: int = 0
    avg_stock_ret_pct: Optional[float] = None
    avg_twii_ret_pct: Optional[float] = None
    avg_inst_net_lots: Optional[float] = None
    note: str = ""


@dataclass
class DynamicFactorReport:
    """底層動態化因子報告：Rolling Beta 權重調校、SRI 風險加速度、自適應 ATR 回撤"""
    rolling_beta_60d: float
    base_w_mri: float
    beta_adj_w_mri: float
    sri_velocity_3d: float
    velocity_tag: str
    velocity_adj: float
    atr_norm_dd: Optional[float] = None
    note: str = ""


@dataclass
class ActiveAlphaReport:
    """主動式進攻因子報告與四象限決策標籤"""
    alpha_score: int
    rating: str
    quadrant: str
    quadrant_badge_cls: str
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
    """v10.2 進階風控與主動因子彙整結果"""
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
# 無前視偏誤向量化因子演算法（v10.2 核心升級）
# =========================================================
def compute_rolling_beta_series(
    stock_close: pd.Series,
    mkt_close: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    clip_bounds: Tuple[float, float] = (0.25, 2.20),
    default_beta: float = 1.0,
) -> pd.Series:
    """
    計算逐日滾動 Beta（無前視偏誤）：
    同步個股與大盤有效報酬遮罩，避免停牌日產生假性 0% 漲跌或分母樣本數不一致。
    """
    aligned_stk = pd.to_numeric(stock_close, errors="coerce").reindex(mkt_close.index)
    s_ret = _safe_pct_change(aligned_stk, periods=1)
    m_ret = _safe_pct_change(pd.to_numeric(mkt_close, errors="coerce"), periods=1)

    both_valid = s_ret.notna() & m_ret.notna()
    s_clean = s_ret.where(both_valid)
    m_clean = m_ret.where(both_valid)

    cov_sm = s_clean.rolling(window=window, min_periods=min_periods).cov(m_clean)
    var_m = m_clean.rolling(window=window, min_periods=min_periods).var()

    beta = cov_sm / var_m.where(var_m >= 1e-10)
    beta = beta.replace([np.inf, -np.inf], np.nan).fillna(default_beta)
    return beta.clip(lower=clip_bounds[0], upper=clip_bounds[1]).round(2).rename("rolling_beta_60d")


def get_dynamic_regime_weights(mri_score: float, rolling_beta: float = 1.0) -> Tuple[float, float, float, str]:
    """
    動態權重狀態機 × 單日 Rolling Beta：
    依當日 MRI 決定基準權重（低波動 15% / 常態 40% / 恐慌 65%），再乘以 Beta 敏感度。
    """
    if pd.isna(mri_score):
        base_mri, regime = 0.40, "常態震盪期"
    elif mri_score < 45.0:
        base_mri, regime = 0.15, "低波動多頭期"
    elif mri_score >= 70.0:
        base_mri, regime = 0.65, "系統性恐慌期"
    else:
        base_mri, regime = 0.40, "常態震盪期"

    beta_val = 1.0 if pd.isna(rolling_beta) else float(rolling_beta)
    beta_mult = float(np.clip(beta_val, 0.55, 1.45))
    w_mri_adj = round(float(np.clip(base_mri * beta_mult, 0.08, 0.78)), 4)
    w_idio_adj = round(1.0 - w_mri_adj, 4)
    label = f"{regime} ({int(round(w_mri_adj * 100))}/{int(round(w_idio_adj * 100))}｜β={beta_val:.2f})"
    return w_mri_adj, w_idio_adj, base_mri, label


def detect_divergence_penalty(
    quality_risk: float,
    outlook_risk: float,
    chip_risk: float,
    trend_risk: float,
) -> Tuple[float, Optional[str]]:
    """
    訊號背離非線性陷阱偵測：
    基本面與展望偏佳（風險分 <= 35）但籌碼與趨勢轉差（風險分 >= 72）時，施加 +15 ~ +20 懲罰。
    """
    fund_avg_risk = (quality_risk + outlook_risk) / 2.0
    action_avg_risk = (chip_risk + trend_risk) / 2.0
    if fund_avg_risk <= 35.0 and action_avg_risk >= 72.0:
        gap = action_avg_risk - fund_avg_risk
        penalty = float(np.clip(15.0 + (gap - 37.0) * 0.2, 15.0, 20.0))
        warning = "⚠️ 偵測到利多背離：基本面佳但法人異常撤出，防範提前反應利空"
        return round(penalty, 1), warning
    return 0.0, None


def compute_chip_pain_series(
    stock_df: pd.DataFrame,
    margin_series: Optional[pd.Series] = None,
    vwap_window: int = 20,
    z_window: int = 60,
    z_min_periods: int = 20,
    margin_lag: int = 0,
) -> pd.DataFrame:
    """
    向量化自適應籌碼痛點指數（無前視偏誤）：
    - 統一採用典型價 (High + Low + Close) / 3 計算 20 日滾動 VWAP
    - 逐日以 60 日滾動 Z-Score 與絕對門檻判斷「多殺多預警 / 籌碼乾淨 / 中性正常」
    """
    idx = stock_df.index
    high = pd.to_numeric(stock_df["High"], errors="coerce")
    low = pd.to_numeric(stock_df["Low"], errors="coerce")
    close = pd.to_numeric(stock_df["Close"], errors="coerce")
    vol = pd.to_numeric(stock_df["Volume"], errors="coerce").fillna(0.0)

    tp = (high + low + close) / 3.0
    vol_sum = vol.rolling(vwap_window, min_periods=vwap_window).sum()
    pv_sum = (tp * vol).rolling(vwap_window, min_periods=vwap_window).sum()
    roll_vwap = (pv_sum / vol_sum.replace(0.0, np.nan)).fillna(
        close.rolling(vwap_window, min_periods=vwap_window).mean()
    )

    price_vs_vwap_pct = (close / roll_vwap.where(roll_vwap > 0.0) - 1.0) * 100.0
    vwap_z = rolling_zscore(price_vs_vwap_pct, window=z_window, min_periods=z_min_periods)

    if margin_series is not None and not margin_series.empty:
        margin_raw = pd.to_numeric(margin_series, errors="coerce")
        margin_raw = margin_raw[~margin_raw.index.duplicated(keep="last")].sort_index()
        margin = margin_raw.reindex(idx)
        if margin_lag > 0:
            margin = margin.shift(margin_lag)
        prev_margin = margin.shift(vwap_window)
        margin_chg_pct = (margin / prev_margin.where(prev_margin > 0.0) - 1.0) * 100.0
        margin_z = rolling_zscore(margin_chg_pct, window=z_window, min_periods=z_min_periods)
    else:
        margin_chg_pct = pd.Series(np.nan, index=idx, dtype=float)
        margin_z = pd.Series(np.nan, index=idx, dtype=float)

    has_data = price_vs_vwap_pct.notna() & margin_chg_pct.notna()
    z_ready = vwap_z.notna() & margin_z.notna()

    is_classic_squeeze = (margin_chg_pct >= 10.0) & (price_vs_vwap_pct <= -8.0)
    is_adaptive_squeeze = z_ready & (margin_z >= 1.5) & (vwap_z <= -1.5) & (price_vs_vwap_pct <= -2.5)
    is_squeeze = has_data & (is_classic_squeeze | is_adaptive_squeeze)

    is_classic_clean = (margin_chg_pct <= -5.0) & (price_vs_vwap_pct >= 0.0)
    is_adaptive_clean = z_ready & (margin_z <= -1.0) & (vwap_z >= 0.5) & (price_vs_vwap_pct >= 0.0)
    is_clean = has_data & (~is_squeeze) & (is_classic_clean | is_adaptive_clean)

    risk_adj = np.select([is_squeeze, is_clean], [15.0, -10.0], default=0.0)
    status = np.select(
        [~has_data, is_squeeze, is_clean],
        ["暫無融資資料", "多殺多預警", "籌碼乾淨"],
        default="中性正常",
    )
    trigger_tag = np.where(
        is_squeeze & (~is_classic_squeeze),
        "自適應異常",
        np.where(is_squeeze, "槓桿套牢", ""),
    )

    return pd.DataFrame(
        {
            "vwap_20": roll_vwap.round(2),
            "price_vs_vwap_pct": price_vs_vwap_pct.round(2),
            "vwap_zscore_60d": vwap_z.round(2),
            "margin_chg_20d_pct": margin_chg_pct.round(1),
            "margin_zscore_60d": margin_z.round(2),
            "pain_status": status,
            "pain_trigger": trigger_tag,
            "pain_risk_adj": risk_adj.astype(float),
            "pain_data_valid": has_data.astype(int),
        },
        index=idx,
    )


def _extract_latest_chip_pain_report(pain_df: pd.DataFrame) -> ChipPainReport:
    """從逐日向量化籌碼痛點表中提取最新一日摘要報告"""
    if pain_df is None or pain_df.empty:
        return ChipPainReport(status="資料不足", risk_adj=0.0, note="交易日不足 20 天")

    last = pain_df.iloc[-1]
    status = str(last.get("pain_status", "中性正常"))
    risk_adj = float(last.get("pain_risk_adj", 0.0))
    vwap_20 = _safe_float(last.get("vwap_20"))
    p_vs_v = _safe_float(last.get("price_vs_vwap_pct"))
    m_chg = _safe_float(last.get("margin_chg_20d_pct"))
    v_z = _safe_float(last.get("vwap_zscore_60d"))
    m_z = _safe_float(last.get("margin_zscore_60d"))
    trigger = str(last.get("pain_trigger", ""))

    z_txt = (
        f"（乖離 Z={v_z:+.1f}σ" + (f" / 融資 Z={m_z:+.1f}σ）" if m_z is not None else "）")
        if v_z is not None else ""
    )

    if status == "暫無融資資料":
        note = f"無法取得完整 20 日融資餘額；現價距20日均價 {_fmt_num(p_vs_v, '%', signed=True)}{z_txt}"
    elif status == "多殺多預警":
        note = (
            f"【{trigger or '槓桿套牢'}】近20日融資 {_fmt_num(m_chg, '%', signed=True)}，"
            f"現價低於20日均價 {_fmt_num(p_vs_v, '%')}{z_txt}，留意斷頭賣壓"
        )
    elif status == "籌碼乾淨":
        note = (
            f"近20日融資減少 {_fmt_num(m_chg, '%', signed=True)} 且站穩20日均價 "
            f"({_fmt_num(p_vs_v, '%', signed=True)}){z_txt}，下檔支撐強"
        )
    else:
        note = (
            f"現價距20日均價 {_fmt_num(p_vs_v, '%', signed=True)}，"
            f"近20日融資變動 {_fmt_num(m_chg, '%', signed=True)}{z_txt}"
        )

    return ChipPainReport(
        status=status,
        risk_adj=risk_adj,
        vwap_20=vwap_20,
        price_vs_vwap_pct=p_vs_v,
        margin_chg_20d_pct=m_chg,
        vwap_zscore_60d=v_z,
        margin_zscore_60d=m_z,
        note=note,
    )


def compute_downside_alpha_series(
    stock_df: pd.DataFrame,
    twii_df: pd.DataFrame,
    inst_net_shares_series: Optional[pd.Series] = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    向量化逐日逆風韌性係數（無前視偏誤）：
    每個交易日 t 僅回看 [t - lookback_days + 1, t] 中「加權指數單日跌逾 1%」之交易日。
    """
    idx = twii_df.index
    if stock_df is None or twii_df is None or stock_df.empty or twii_df.empty:
        return pd.DataFrame(
            {
                "down_days_count": 0,
                "avg_stock_ret_pct": np.nan,
                "avg_twii_ret_pct": np.nan,
                "avg_inst_net_lots": np.nan,
                "has_badge": 0,
                "resilience_bonus": 0.0,
            },
            index=idx,
        )

    s_close = pd.to_numeric(stock_df["Close"], errors="coerce").reindex(idx)
    m_close = pd.to_numeric(twii_df["Close"], errors="coerce")
    s_ret = _safe_pct_change(s_close, periods=1) * 100.0
    m_ret = _safe_pct_change(m_close, periods=1) * 100.0

    is_down = ((m_ret <= -1.0) & s_ret.notna() & m_ret.notna()).astype(float)
    down_cnt = is_down.rolling(lookback_days, min_periods=1).sum()
    safe_cnt = down_cnt.where(down_cnt > 0.0)

    avg_s = (s_ret.where(is_down == 1.0, 0.0)).rolling(lookback_days, min_periods=1).sum() / safe_cnt
    avg_m = (m_ret.where(is_down == 1.0, 0.0)).rolling(lookback_days, min_periods=1).sum() / safe_cnt

    if inst_net_shares_series is not None and not inst_net_shares_series.empty:
        inst_raw = pd.to_numeric(inst_net_shares_series, errors="coerce")
        inst_raw = inst_raw[~inst_raw.index.duplicated(keep="last")].sort_index()
        inst_lots = inst_raw.reindex(idx) / 1000.0
        inst_valid_down = ((is_down == 1.0) & inst_lots.notna()).astype(float)
        inst_cnt = inst_valid_down.rolling(lookback_days, min_periods=1).sum()
        avg_inst = (
            inst_lots.where(inst_valid_down == 1.0, 0.0).rolling(lookback_days, min_periods=1).sum()
            / inst_cnt.where(inst_cnt > 0.0)
        )
    else:
        avg_inst = pd.Series(np.nan, index=idx, dtype=float)

    cond_pos_ret = avg_s > 0.0
    cond_inst_defend = avg_inst.notna() & (avg_inst > 0.0) & ((avg_s - avg_m) >= 1.0)
    has_badge = ((down_cnt >= 2.0) & (cond_pos_ret | cond_inst_defend)).astype(int)
    resilience_bonus = np.where(has_badge == 1, -5.0, 0.0)

    return pd.DataFrame(
        {
            "down_days_count": down_cnt.fillna(0).astype(int),
            "avg_stock_ret_pct": avg_s.round(2),
            "avg_twii_ret_pct": avg_m.round(2),
            "avg_inst_net_lots": avg_inst.round(1),
            "has_badge": has_badge,
            "resilience_bonus": resilience_bonus.astype(float),
        },
        index=idx,
    )


def _extract_latest_downside_report(downside_df: pd.DataFrame) -> DownsideAlphaReport:
    """從逐日逆風韌性表中提取最新一日摘要報告"""
    if downside_df is None or downside_df.empty:
        return DownsideAlphaReport(has_badge=False, note="大盤或個股歷史價格不足")

    last = downside_df.iloc[-1]
    cnt = int(last.get("down_days_count", 0))
    if cnt == 0:
        return DownsideAlphaReport(
            has_badge=False,
            down_days_count=0,
            note="近 60 日無大盤單日跌逾 1% 之交易日",
        )

    avg_s = _safe_float(last.get("avg_stock_ret_pct"))
    avg_m = _safe_float(last.get("avg_twii_ret_pct"))
    avg_inst = _safe_float(last.get("avg_inst_net_lots"))
    has_badge = bool(int(last.get("has_badge", 0)) == 1)

    inst_txt = f"，法人逆風日均買超 {avg_inst:+.0f} 張" if avg_inst is not None else ""
    note = (
        f"近60日大盤大跌共 {cnt} 天（均跌 {_fmt_num(avg_m, '%')}），"
        f"本股均表現 {_fmt_num(avg_s, '%', signed=True)}{inst_txt}"
    )
    return DownsideAlphaReport(
        has_badge=has_badge,
        down_days_count=cnt,
        avg_stock_ret_pct=avg_s,
        avg_twii_ret_pct=avg_m,
        avg_inst_net_lots=avg_inst,
        note=note,
    )


def evaluate_sri_velocity(base_sri_history: List[float]) -> Tuple[float, str, float]:
    """
    風險加速度 (SRI Velocity / 二階斜率)：
    追蹤近 3 個交易日的風險變化速度並給予提前預警或正向收斂微調。
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
    主動式進攻因子 (Active Alpha Score) 與四象限分類：
    結合相對大盤強弱動能 (RS)、波動率壓縮突破 (ATR Squeeze) 與法人低檔吸納轉折。
    """
    if df is None or len(df) < 25:
        return ActiveAlphaReport(
            alpha_score=50,
            rating="溫和蓄勢",
            quadrant="🛡️ 防禦盤整",
            quadrant_badge_cls="badge-blue",
            quadrant_desc="歷史交易日不足，暫以中性基準顯示",
            signals=["交易日資料尚不足以計算完整主動動能"],
        )

    last = df.iloc[-1]
    score = 45.0
    signals: List[str] = []

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
        if cmf_10 >= 0.14:
            score += 10.0
            signals.append("價量資金流向 (CMF) 呈現主動淨流入")
        elif cmf_10 <= -0.14:
            score -= 10.0
            signals.append("價量資金流向 (CMF) 呈現資金流出")

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
BEARISH_KW: Dict[str, float] = {
    "升息": 3.0, "通膨": 2.5, "關稅": 3.0, "制裁": 2.5, "禁令": 2.5,
    "重挫": 3.0, "崩跌": 3.5, "暴跌": 3.5, "斷頭": 3.5, "衰退": 3.0,
    "停滯": 2.5, "戰爭": 3.0, "衝突": 2.5, "賣超": 2.0, "提款": 2.0,
    "緊縮": 2.5, "鷹派": 2.5, "下修": 2.0, "恐慌": 2.5, "破底": 2.5,
    "大跌": 2.5, "利空": 2.0, "匯出": 2.0,
}
BULLISH_KW: Dict[str, float] = {
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
    """封裝 yfinance 歷史行情擷取，統一去重排序與無時區日期索引"""
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _dedup_sort_index(df)
    except Exception:
        return pd.DataFrame()


def _finmind_get(
    dataset: str,
    start: str,
    end: str,
    token: str,
    data_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
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
    macro_data: Dict[str, pd.DataFrame] = {}
    for k, sym in macro_tickers.items():
        d = _fetch_yf(sym, fetch_start, end)
        if k == "TNX" and not d.empty and "Close" in d and d["Close"].median() > 20:
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
            s = s[~s.index.duplicated(keep="last")].sort_index()
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
            frames.append(_dedup_sort_index(piv[["Foreign_Investor"]].rename(columns={"Foreign_Investor": "mkt_foreign_net_b"})))

    mkt_chips = _dedup_sort_index(pd.concat(frames, axis=1)) if frames else pd.DataFrame()
    return twii, macro_data, mkt_chips, mkt_status


def fetch_fundamentals_and_outlook_smart(
    clean_id: str,
    resolved_sym: str,
    latest_close: float,
    latest_dd_20d: float,
    token: str,
) -> Tuple[FundamentalQualityReport, ForwardOutlookReport, str]:
    """
    智慧基本面＋未來展望擷取（Yahoo Finance + FinMind 雙重備援）：
    - 修正賣方目標價樂觀偏誤，股價深跌時自動折價防範滯後陷阱
    - 僅在成功抓到真實數據時寫入 session_state 快取 30 分鐘
    """
    if "fund_outlook_cache" not in st.session_state:
        st.session_state["fund_outlook_cache"] = {}
    cache_key = f"{clean_id}_{bool(token)}_v102"
    cached = st.session_state["fund_outlook_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 1800):
        return cached["quality"], cached["outlook"], cached["yf_name"]

    pe = pb = roe = op_m = rev_g = None
    trailing_pe = forward_pe = target_mean = None
    analyst_cnt = None
    rev_accel = None
    yf_name = ""
    used_sources: List[str] = []

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
    q_strengths: List[str] = []
    q_warns: List[str] = []
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
        quality_score=q_score,
        risk_equivalent=float(100 - q_score),
        rating=q_rating,
        pe=pe,
        pb=pb,
        roe_pct=roe,
        op_margin_pct=op_m,
        rev_growth_pct=rev_g,
        source_note=" + ".join(used_sources) if used_sources else "暫無資料",
        strengths=q_strengths if q_strengths else ["財務結構大致符合產業平均水準"],
        warnings=q_warns,
    )

    # 評分 B：未來展望與法人預期（修正賣方樂觀偏誤、極端值封頂與深跌現實鎖定）
    o_score = 45.0
    o_strengths: List[str] = []
    o_warns: List[str] = []
    implied_eps_g = None
    target_upside = None
    is_deep_pullback = pd.notna(latest_dd_20d) and latest_dd_20d <= -12.0
    is_severe_crash = pd.notna(latest_dd_20d) and latest_dd_20d <= -15.0
    strong_eps = strong_tgt = strong_rev = False

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

    if target_mean is not None and latest_close > 0:
        target_upside = round((target_mean / latest_close - 1.0) * 100.0, 1)
        cnt_txt = f"({analyst_cnt}家機構)" if analyst_cnt else ""
        if target_upside >= 10.0 and is_deep_pullback:
            o_warns.append(
                f"法人目標價 {target_mean:.1f} 元{cnt_txt} 雖顯示 {target_upside:+.1f}% 空間，"
                f"但近月已回檔 {latest_dd_20d:.1f}%，防範滯後下修（暫不計入加分）"
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

    if strong_eps and strong_tgt and strong_rev and pd.notna(latest_dd_20d) and latest_dd_20d > -8.0:
        o_score += 6.0
        o_strengths.append("獲利預估、法人目標與營收加速三因子健康共振")

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
        outlook_score=o_score_int,
        risk_equivalent=float(100 - o_score_int),
        rating=o_rating,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        implied_eps_growth_pct=implied_eps_g,
        target_mean_price=target_mean,
        target_upside_pct=target_upside,
        analyst_count=analyst_cnt,
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
        df_m = df_m[~df_m.index.duplicated(keep="last")].sort_index()
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
        piv = _dedup_sort_index(piv)
        i_out = pd.DataFrame(index=piv.index)
        i_out["foreign_net"] = piv["Foreign_Investor"] if "Foreign_Investor" in piv else 0.0
        i_out["trust_net"] = piv["Investment_Trust"] if "Investment_Trust" in piv else 0.0
        i_out["smart_money_net"] = i_out["foreign_net"] + i_out["trust_net"]
        frames.append(i_out)

    stk_chips = _dedup_sort_index(pd.concat(frames, axis=1)) if frames else pd.DataFrame()
    return stk, resolved_sym, clean_id, stk_chips, stk_status


# =========================================================
# v10.2 核心量化風控＋主動式雙軸運算引擎（無前視偏誤版）
# =========================================================
class InstitutionalQuantRiskEngine:
    """台股量化風控與主動進攻雙軸引擎（v10.2：逐日滾動 Beta / 籌碼痛點 / 逆風韌性 / 美股跨日 PIT 對齊）"""

    US_SHIFT_KEYS = {"SOX", "NDX", "TNX", "GOLD", "OIL", "VIX"}

    def __init__(self, config: Optional[RiskModelConfig] = None):
        self.cfg = config or RiskModelConfig()
        self.w = self.cfg.weights
        self.mri_weights = np.arange(1, self.cfg.window_days + 1, dtype=float)
        self.chip_weights = np.arange(1, self.cfg.chip_smooth_days + 1, dtype=float)

    def score_market_snapshot(self, row: pd.Series, news_risk_score: float = 50.0) -> float:
        """計算單日大盤系統風險快照分數（0~100）"""
        s_us = 45.0
        sox_5d, sox_10d = row.get("sox_5d_pct", np.nan), row.get("sox_10d_pct", np.nan)
        ndx_5d, ndx_10d = row.get("ndx_5d_pct", np.nan), row.get("ndx_10d_pct", np.nan)
        if pd.notna(sox_10d):
            if sox_10d <= -8.0 or (pd.notna(sox_5d) and sox_5d <= -5.5):
                s_us += 38
            elif sox_10d <= -4.0 or (pd.notna(sox_5d) and sox_5d <= -3.0):
                s_us += 22
            elif sox_10d >= 5.0 and pd.notna(ndx_10d) and ndx_10d >= 2.5:
                s_us -= 20
        if pd.notna(ndx_10d):
            if ndx_10d <= -4.5 or (pd.notna(ndx_5d) and ndx_5d <= -3.0):
                s_us += 18
            elif ndx_10d >= 3.0:
                s_us -= 8
        s_us = float(np.clip(s_us, 5, 100))

        s_rf = 42.0
        tnx, tnx_20d = row.get("tnx_close", np.nan), row.get("tnx_20d_chg", np.nan)
        twd_10d = row.get("usdtwd_10d_pct", np.nan)
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
        bias_60 = row.get("twii_bias_60", np.nan)
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
            twii_dd, twii_hv = row.get("twii_dd_60d", np.nan), row.get("twii_hv_20", np.nan)
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
        oil_10d, oil_c = row.get("oil_10d_pct", np.nan), row.get("oil_close", np.nan)
        gold_10d = row.get("gold_10d_pct", np.nan)
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

        s_btc = 45.0
        btc_10d = row.get("btc_10d_pct", np.nan)
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

        s_vix = 42.0
        vix, vix_5d = row.get("vix_close", np.nan), row.get("vix_5d_pct", np.nan)
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
            s_us * self.w.w_us_tech
            + s_rf * self.w.w_rates_fx
            + s_tw * self.w.w_tw_market
            + s_cmd * self.w.w_commodities
            + s_btc * self.w.w_btc_liquidity
            + s_sent * self.w.w_vix_news
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
        """計算自適應技術趨勢風險分（以個股自身 ATR 正規化乖離與回撤）"""
        close_p = float(row.get("known_stk_close", 0.0))
        vwap_20 = _safe_float(row.get("stk_vwap_20"))
        if vwap_20 is None or vwap_20 <= 0:
            vwap_20 = close_p
        atr = _safe_float(row.get("stk_atr_14"))
        if atr is None or atr <= 0:
            atr = max(close_p * 0.025, 0.01)
        atr_dd = _safe_float(row.get("stk_atr_norm_dd"))

        t_risk = 45.0
        vwap_atr_diff = (close_p - vwap_20) / atr
        if vwap_atr_diff < 0:
            t_risk += min(abs(vwap_atr_diff) * 9.5, 30.0)
        else:
            t_risk -= min(vwap_atr_diff * 5.0, 12.0)

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
        high_15 = _safe_float(row.get("stk_high_15d")) or close_p
        atr = _safe_float(row.get("stk_atr_14")) or (close_p * 0.025)
        vwap_20 = _safe_float(row.get("stk_vwap_20")) or close_p
        if atr <= 0:
            atr = close_p * 0.025
        eff_atr = min(atr, close_p * 0.065)
        if vwap_20 <= 0:
            vwap_20 = close_p
        if high_15 < close_p:
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
        """執行完整單股或批量雙軸分析（全歷史序列無前視偏誤）"""
        today = datetime.date.today()
        fetch_start = (today - datetime.timedelta(days=450)).strftime("%Y-%m-%d")
        fm_start = (today - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        end_str = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        token = get_finmind_token()

        twii, macro_data, mkt_chips, mkt_status = fetch_shared_market_data(fetch_start, end_str, fm_start, token)
        stk, resolved_sym, clean_id, stk_chips, stk_status = fetch_single_stock_price_and_chips(
            stock_id, fetch_start, end_str, fm_start, token
        )
        if stk.empty or twii.empty or "Close" not in stk.columns or "Close" not in twii.columns:
            raise ValueError(f"查無股票代號「{stock_id}」，請確認是否為有效的台股上市／上櫃代碼。")

        stk_close_clean = pd.to_numeric(stk["Close"], errors="coerce").dropna()
        if stk_close_clean.empty:
            raise ValueError(f"股票代號「{stock_id}」缺乏有效收盤價資料。")

        latest_close_val = float(stk_close_clean.iloc[-1])
        tail_20_max = float(stk_close_clean.tail(20).max())
        latest_dd_20d = float((latest_close_val / tail_20_max - 1.0) * 100.0) if tail_20_max > 0 else 0.0

        quality, outlook, yf_name = fetch_fundamentals_and_outlook_smart(
            clean_id, resolved_sym, latest_close_val, latest_dd_20d, token
        )
        name_map = fetch_tw_stock_name_map(token)
        stock_name = name_map.get(clean_id) or yf_name
        display_name = f"{clean_id}-{stock_name}" if stock_name else clean_id

        live_news = fetch_news_risk_score("") if fast_batch else fetch_news_risk_score(stock_id)
        mkt_note = "真實籌碼" if not mkt_chips.empty else f"備援({mkt_status})"
        stk_note = "真實籌碼" if not stk_chips.empty else f"資金流代理({stk_status})"

        # 對齊大盤與個股日線序列
        df = pd.DataFrame(index=twii.index)
        twii_c = pd.to_numeric(twii["Close"], errors="coerce")
        stk_h = pd.to_numeric(stk["High"], errors="coerce").reindex(df.index).ffill()
        stk_l = pd.to_numeric(stk["Low"], errors="coerce").reindex(df.index).ffill()
        stk_c = pd.to_numeric(stk["Close"], errors="coerce").reindex(df.index).ffill()
        stk_v = pd.to_numeric(stk["Volume"], errors="coerce").reindex(df.index).fillna(0.0)

        twii_ma60 = twii_c.rolling(60).mean()
        twii_ma240 = twii_c.rolling(240, min_periods=120).mean()
        df["twii_bias_60"] = (twii_c / twii_ma60 - 1.0) * 100.0
        df["twii_below_ma60"] = (twii_c < twii_ma60).astype(int)
        df["twii_below_ma240"] = (twii_c < twii_ma240).astype(int)
        df["twii_hv_20"] = _safe_pct_change(twii_c, 1).rolling(20).std() * np.sqrt(252) * 100.0
        df["twii_dd_60d"] = (twii_c / twii_c.rolling(60, min_periods=10).max() - 1.0) * 100.0

        prev_c = stk_c.shift(1)
        tr = pd.concat([(stk_h - stk_l), (stk_h - prev_c).abs(), (stk_l - prev_c).abs()], axis=1).max(axis=1)
        df["stk_atr_14"] = tr.rolling(14).mean()
        df["stk_atr_10"] = tr.rolling(10).mean()
        df["stk_atr_60"] = tr.rolling(60, min_periods=20).mean()
        df["vol_squeeze_ratio"] = df["stk_atr_10"] / df["stk_atr_60"].replace(0.0, np.nan)
        df["min_sq_5d"] = df["vol_squeeze_ratio"].rolling(5, min_periods=1).min()

        df["stk_high_15d"] = stk_h.rolling(15).max()
        df["stk_high_20d"] = stk_h.rolling(20, min_periods=5).max()
        df["stk_atr_norm_dd"] = (df["stk_high_20d"] - stk_c) / df["stk_atr_14"].replace(0.0, np.nan)
        df["stk_vol_ratio_20"] = stk_v / stk_v.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
        df["stk_ret_10d"] = _safe_pct_change(stk_c, 10) * 100.0

        # 相對大盤強弱指標 (RS Momentum)
        rs_line = stk_c / twii_c.replace(0.0, np.nan)
        rs_ma20 = rs_line.rolling(20).mean()
        rs_high60 = rs_line.rolling(60, min_periods=20).max()
        df["rs_vs_ma20_pct"] = (rs_line / rs_ma20 - 1.0) * 100.0
        df["rs_new_high_60d"] = (rs_line >= rs_high60 * 0.995).astype(int)

        hl_range = (stk_h - stk_l).replace(0.0, np.nan)
        mfm = (((stk_c - stk_l) - (stk_h - stk_c)) / hl_range).fillna(0.0)
        df["stk_cmf_10"] = (mfm * stk_v).rolling(10).sum() / stk_v.rolling(10).sum().replace(0.0, np.nan)
        df["stk_dd_20d"] = (stk_c / stk_c.rolling(20, min_periods=5).max() - 1.0) * 100.0

        # 總經資料 PIT 對齊：美股收盤日 +1 天後才對台股生效，消除跨時區前視偏誤
        def align_macro(k: str) -> pd.Series:
            d_macro = macro_data.get(k)
            if d_macro is None or d_macro.empty or "Close" not in d_macro.columns:
                return pd.Series(np.nan, index=df.index, dtype=float)
            s = pd.to_numeric(d_macro["Close"], errors="coerce").dropna()
            if s.empty:
                return pd.Series(np.nan, index=df.index, dtype=float)
            if k in self.US_SHIFT_KEYS:
                s = s.copy()
                s.index = s.index + pd.Timedelta(days=1)
            return s.reindex(df.index, method="ffill")

        sox, ndx, tnx = align_macro("SOX"), align_macro("NDX"), align_macro("TNX")
        df["sox_5d_pct"], df["sox_10d_pct"] = _safe_pct_change(sox, 5) * 100.0, _safe_pct_change(sox, 10) * 100.0
        df["ndx_5d_pct"], df["ndx_10d_pct"] = _safe_pct_change(ndx, 5) * 100.0, _safe_pct_change(ndx, 10) * 100.0
        df["tnx_close"], df["tnx_20d_chg"] = tnx, tnx - tnx.shift(20)
        df["usdtwd_10d_pct"] = _safe_pct_change(align_macro("USDTWD"), 10) * 100.0
        df["gold_10d_pct"] = _safe_pct_change(align_macro("GOLD"), 10) * 100.0
        oil = align_macro("OIL")
        df["oil_close"], df["oil_10d_pct"] = oil, _safe_pct_change(oil, 10) * 100.0
        df["btc_10d_pct"] = _safe_pct_change(align_macro("BTC"), 10) * 100.0
        vix = align_macro("VIX")
        df["vix_close"], df["vix_5d_pct"] = vix, _safe_pct_change(vix, 5) * 100.0

        if not mkt_chips.empty:
            mc = mkt_chips.reindex(df.index).ffill()
            if "mkt_margin_bal" in mc:
                df["mkt_margin_rank_60d"] = mc["mkt_margin_bal"].rolling(60, min_periods=20).rank(pct=True)
                df["mkt_margin_5d_pct"] = _safe_pct_change(mc["mkt_margin_bal"], 5) * 100.0
            if "mkt_foreign_net_b" in mc:
                df["mkt_foreign_5d_b"] = mc["mkt_foreign_net_b"].rolling(5).sum()
                df["mkt_foreign_10d_b"] = mc["mkt_foreign_net_b"].rolling(10).sum()

        if not stk_chips.empty:
            sc = stk_chips.reindex(df.index)
            if "margin_bal" in sc:
                m_bal = sc["margin_bal"].ffill()
                df["margin_5d_chg_pct"] = _safe_pct_change(m_bal, 5) * 100.0
                df["margin_10d_chg_pct"] = _safe_pct_change(m_bal, 10) * 100.0
                df["margin_util_pct"] = sc["margin_util_pct"].ffill() if "margin_util_pct" in sc else np.nan
                df["short_margin_ratio"] = sc["short_margin_ratio"].ffill() if "short_margin_ratio" in sc else 0.0
            if "smart_money_net" in sc:
                sm_net = sc["smart_money_net"].fillna(0.0)
                tr_net = sc["trust_net"].fillna(0.0) if "trust_net" in sc else pd.Series(0.0, index=df.index)
                v5 = stk_v.rolling(5).sum().replace(0.0, np.nan)
                v10 = stk_v.rolling(10).sum().replace(0.0, np.nan)
                df["smart_5d_vol_ratio"] = (sm_net.rolling(5).sum() / v5) * 100.0
                df["smart_prev5d_vol_ratio"] = df["smart_5d_vol_ratio"].shift(5)
                df["smart_10d_vol_ratio"] = (sm_net.rolling(10).sum() / v10) * 100.0
                is_sell = (tr_net < 0).astype(int)
                is_buy = (tr_net > 0).astype(int)
                df["trust_consec_sell"] = np.where(
                    is_sell == 1, is_sell.groupby((is_sell != is_sell.shift()).cumsum()).cumsum(), 0
                )
                df["trust_consec_buy"] = np.where(
                    is_buy == 1, is_buy.groupby((is_buy != is_buy.shift()).cumsum()).cumsum(), 0
                )
                df["trust_prev15_net"] = tr_net.shift(3).rolling(15).sum()

        # 逐日向量化計算無前視偏誤之 Rolling Beta、籌碼痛點與逆風韌性
        beta_series = compute_rolling_beta_series(stk["Close"], twii["Close"], window=60, min_periods=20)
        margin_s = stk_chips["margin_bal"] if (not stk_chips.empty and "margin_bal" in stk_chips) else None
        inst_s = stk_chips["smart_money_net"] if (not stk_chips.empty and "smart_money_net" in stk_chips) else None

        pain_df_raw = compute_chip_pain_series(stk, margin_s, vwap_window=20, z_window=60, z_min_periods=20, margin_lag=0)
        downside_df = compute_downside_alpha_series(stk, twii, inst_s, lookback_days=60)

        # 統一將典型價 20 日 VWAP、每日籌碼痛點與每日逆風加分對齊回大盤交易日曆
        df["stk_vwap_20"] = pain_df_raw["vwap_20"].reindex(df.index).ffill()
        df["rolling_beta_60d"] = beta_series.reindex(df.index).ffill().fillna(1.0)
        df["pain_risk_adj"] = pain_df_raw["pain_risk_adj"].reindex(df.index).ffill().fillna(0.0)
        df["resilience_bonus"] = downside_df["resilience_bonus"].reindex(df.index).fillna(0.0)

        pain_report = _extract_latest_chip_pain_report(pain_df_raw)
        downside_report = _extract_latest_downside_report(downside_df)

        work_df = df.dropna(subset=["stk_atr_14"]).tail(lookback_days + 15).copy()
        work_df["known_stk_close"] = stk_c.reindex(work_df.index)
        work_df = work_df.dropna(subset=["known_stk_close"])
        n = len(work_df)
        if n == 0:
            raise ValueError(f"股票代號「{stock_id}」歷史交易日數不足，無法完成風控運算。")

        mri_snaps = np.array(
            [
                self.score_market_snapshot(work_df.iloc[i], news_risk_score=(live_news if i == n - 1 else 50.0))
                for i in range(n)
            ],
            dtype=float,
        )
        chip_snaps = np.array([self.score_stock_chips_snapshot(work_df.iloc[i]) for i in range(n)], dtype=float)

        combined_q = int(round(quality.quality_score * 0.5 + outlook.outlook_score * 0.5))
        history = []
        base_sri_seq: List[float] = []
        latest_overlay: Optional[AdvancedRiskOverlay] = None
        w_m, w_c = self.cfg.window_days, self.cfg.chip_smooth_days

        for i in range(max(0, n - lookback_days), n):
            is_last = (i == n - 1)
            m_win = mri_snaps[max(0, i - w_m + 1) : i + 1]
            c_win = chip_snaps[max(0, i - w_c + 1) : i + 1]
            row = work_df.iloc[i]

            wma_10 = float(np.average(m_win, weights=self.mri_weights[-len(m_win):]))
            mom = float(np.mean(m_win[-3:]) - np.mean(m_win[:-3])) if len(m_win) >= 6 else 0.0
            trend_adj = min(mom * 0.45, 13.0) if mom >= 5.0 else (max(mom * 0.20, -6.0) if mom <= -7.0 else 0.0)
            high_days = int(np.sum(m_win >= 65.0))
            persist_adj = 7.0 if high_days >= 7 else (3.5 if high_days >= 5 else 0.0)
            mri_score = int(np.clip(round(max(wma_10 + trend_adj + persist_adj, self._get_lock_floor(row))), 0, 100))

            # 嚴格使用當日滾動 Beta、當日籌碼痛點與當日逆風韌性加分（無前視偏誤）
            day_beta = float(row.get("rolling_beta_60d", 1.0))
            day_pain_adj = float(row.get("pain_risk_adj", 0.0))
            day_res_bonus = float(row.get("resilience_bonus", 0.0))

            w_mri_dyn, w_idio_dyn, base_w_mri, regime_name = get_dynamic_regime_weights(mri_score, day_beta)

            chip_smooth = float(np.average(c_win, weights=self.chip_weights[-len(c_win):]))
            adj_chip = float(np.clip(chip_smooth + day_pain_adj, 5.0, 98.0))
            idio = (
                adj_chip * self.w.w_stock_chip_in_idio
                + quality.risk_equivalent * self.w.w_stock_health_in_idio
                + outlook.risk_equivalent * self.w.w_stock_outlook_in_idio
            )

            trend_risk = self.score_stock_trend_risk(row)
            div_penalty, div_warn = detect_divergence_penalty(
                quality_risk=quality.risk_equivalent,
                outlook_risk=outlook.risk_equivalent,
                chip_risk=adj_chip,
                trend_risk=trend_risk,
            )

            base_sri = mri_score * w_mri_dyn + idio * w_idio_dyn + div_penalty + day_res_bonus
            if adj_chip > 68.0:
                base_sri += min((adj_chip - 68.0) * 0.45, 10.0)
            base_sri_seq.append(base_sri)

            vel_3d, vel_tag, vel_adj = evaluate_sri_velocity(base_sri_seq)
            sri_score = int(np.clip(round(base_sri + vel_adj), 0, 100))

            if is_last:
                atr_dd_val = _safe_float(row.get("stk_atr_norm_dd"))
                dyn_note = (
                    f"60日 Beta={day_beta:.2f}（大盤權重由基準 {int(base_w_mri * 100)}% 動態調整為 {int(round(w_mri_dyn * 100))}%）；"
                    f"近3日風險斜率 {vel_3d:+.1f} 分（{vel_tag}）"
                )
                dyn_report = DynamicFactorReport(
                    rolling_beta_60d=day_beta,
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
            history.append(
                {
                    "date": work_df.index[i],
                    "close": round(float(row["known_stk_close"]), 2),
                    "vwap_20": round(float(row["stk_vwap_20"]), 2) if pd.notna(row.get("stk_vwap_20")) else np.nan,
                    "mri": mri_score,
                    "sri": sri_score,
                    "chip": int(round(adj_chip)),
                    "sell_p": sell_p,
                    "stop_p": stop_p,
                }
            )

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
def _metric_tag(val, good_cond, bad_cond, good_txt: str, mid_txt: str, bad_txt: str, empty_txt: str = "無資料") -> str:
    """依數值條件回傳對應顏色與文字的指標狀態標籤"""
    if val is None or pd.isna(val):
        return f'<span class="m-tag" style="color:#6e7681;">{empty_txt}</span>'
    if good_cond(val):
        return f'<span class="m-tag tag-good">{good_txt}</span>'
    if bad_cond(val):
        return f'<span class="m-tag tag-bad">{bad_txt}</span>'
    return f'<span class="m-tag tag-mid">{mid_txt}</span>'


def render_active_alpha_card(aa: ActiveAlphaReport) -> None:
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

    _render_html(
        f"""<div class="risk-card">
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
        </div>"""
    )


def render_advanced_overlay_card(ov: AdvancedRiskOverlay) -> None:
    """渲染動態量化風控偵測卡片"""
    dyn = ov.dynamic_factors
    vel_cls = (
        "badge-red" if dyn.velocity_tag == "加速惡化" else
        ("badge-yellow" if dyn.velocity_tag == "緩步升溫" else
         ("badge-green" if dyn.velocity_tag == "風險收斂" else "badge-blue"))
    )

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

    _render_html(
        f"""<div class="risk-card">
        <div class="card-title">🧠 動態風控引擎偵測（Beta 狀態機 / 風險加速度 / Z-Score 痛點 / 逆風韌性）</div>
        <div style="margin:6px 0 8px 0;">{badges_html}</div>
        {div_html}
        <div class="health-list">
        <div>• <b>動態調校：</b>{dyn.note}</div>
        <div>• <b>籌碼痛點 (20日VWAP)：</b>{ov.pain_report.note}</div>
        <div>• <b>逆風韌性 (Downside Alpha)：</b>{ov.downside_alpha.note}</div>
        </div>
        </div>"""
    )


def render_health_card(q: FundamentalQualityReport) -> None:
    """渲染歷史財務體質健檢卡片"""
    roe_t = _metric_tag(q.roe_pct, lambda x: x >= 15, lambda x: x < 5, "優異", "穩健", "偏弱")
    op_t = _metric_tag(q.op_margin_pct, lambda x: x >= 15, lambda x: x < 3, "高毛利", "普通", "偏薄")
    rev_t = _metric_tag(q.rev_growth_pct, lambda x: x >= 15, lambda x: x <= -10, "高成長", "平穩", "衰退")
    pe_t = _metric_tag(q.pe, lambda x: 0 < x <= 16, lambda x: x >= 38 or x < 0, "便宜", "合理", "偏貴/虧損")
    pb_t = _metric_tag(q.pb, lambda x: 0 < x <= 2.0, lambda x: x >= 6.0, "低估", "合理", "溢價")
    pb_str = f"{q.pb:.2f}x" if q.pb is not None else "N/A"
    str_html = "".join([f"<div>✅ {s}</div>" for s in q.strengths])
    warn_html = "".join([f"<div>⚠️ {w}</div>" for w in q.warnings]) if q.warnings else "<div>✨ 目前無明顯財務警訊</div>"
    _render_html(
        f"""<div class="risk-card">
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
        </div>"""
    )


def render_outlook_card(o: ForwardOutlookReport) -> None:
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
    _render_html(
        f"""<div class="risk-card">
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
        </div>"""
    )


def render_dark_chart(res: Dict) -> None:
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
# 側邊欄：FinMind Token 設定與 v10.2 模型說明
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
        "v10.2 無前視偏誤升級：\n"
        "• 逐日滾動 60 日 Beta 狀態機\n"
        "• 逐日滾動 20 日典型價 VWAP 痛點\n"
        "• 逐日滾動 60 日逆風抗跌係數\n"
        "• 美股總經 +1 日 PIT 時間對齊\n\n"
        "四象限定位（SRI 52 / Alpha 60）：\n"
        "• 🚀 主升段首選：低險＋高攻\n"
        "• 🛡️ 防禦盤整：低險＋低攻\n"
        "• ⚡ 高檔博弈：高險＋高攻\n"
        "• 🚫 弱勢迴避：高險＋低攻"
    )


# =========================================================
# 主介面：雙分頁（單股雙軸診斷 / 多股批量掃描）
# =========================================================
st.markdown("## 🛡️ 台股量化風控儀表板 v10.2")
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

                _render_html(
                    f"### {res['display_name']} &nbsp; <span class='{aa.quadrant_badge_cls}' style='vertical-align:middle;'>{aa.quadrant}</span>"
                )
                st.caption(
                    f"基準日：{res['date']} ｜ 收盤價：{res['close']} 元 ｜ 60日 Beta：{dyn.rolling_beta_60d:.2f} ｜ 籌碼：{res['stk_note']}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    _render_html(
                        f"""<div class="risk-card">
                        <div class="card-title">個股風險指數 (SRI)</div>
                        <div class="card-value" style="color:{s_col};">{s_icon} {res['sri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
                        <div class="card-sub"><span class="{s_cls}">{s_stat}</span> 3日斜率 {dyn.sri_velocity_3d:+.1f}｜籌碼 {res['chip']}</div>
                        </div>"""
                    )
                with c2:
                    _render_html(
                        f"""<div class="risk-card">
                        <div class="card-title">大盤風險指數 (MRI)</div>
                        <div class="card-value" style="color:{m_col};">{m_icon} {res['mri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
                        <div class="card-sub"><span class="{m_cls}">{m_stat}</span> Beta權重 {mri_weight_pct}%</div>
                        </div>"""
                    )

                c3, c4 = st.columns(2)
                with c3:
                    _render_html(
                        f"""<div class="risk-card">
                        <div class="card-title">建議賣出價（目標調節區）</div>
                        <div class="card-value" style="color:#d2a8ff;">{res['sell_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
                        <div class="card-sub">距收盤 {sell_pct:+.1f}%</div>
                        </div>"""
                    )
                with c4:
                    _render_html(
                        f"""<div class="risk-card">
                        <div class="card-title">下檔防守價</div>
                        <div class="card-value" style="color:#f0883e;">{res['stop_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
                        <div class="card-sub">距收盤 {stop_pct:+.1f}%</div>
                        </div>"""
                    )

                render_active_alpha_card(aa)
                render_advanced_overlay_card(ov)
                render_outlook_card(res["outlook"])
                render_health_card(res["quality"])
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

                    rows.append(
                        {
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
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
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
                        }
                    )
                prog.progress((idx + 1) / len(tickers))
            prog.empty()
            status_txt.empty()
            st.session_state["batch_results"] = pd.DataFrame(rows)

    if "batch_results" in st.session_state and not st.session_state["batch_results"].empty:
        st.dataframe(st.session_state["batch_results"], use_container_width=True, hide_index=True)

# 底部免責聲明
_render_html(
    '<div class="disclaimer">本系統所有數據與風險指標皆由公開資訊與量化演算法自動運算產生，僅供風險控管參考，絕非投資買賣建議。<br>投資人應獨立判斷並自負盈虧與法律責任。</div>'
)
