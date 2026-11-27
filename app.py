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
=========================================================
#頁面基本設定與黑底質感 CSS（手機優先排版）
=========================================================
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
內建熱門台股中文名稱備援表（當 API 限流時仍可正確顯示中文）
FALLBACK_STOCK_NAMES = {
"2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2382": "廣達", "2308": "台達電",
"2303": "聯電", "3711": "日月光投控", "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
"2886": "兆豐金", "2884": "玉山金", "2383": "台光電", "6274": "台燿", "8046": "南電",
"1815": "富喬", "3231": "緯創", "6669": "緯穎", "2357": "華碩", "2376": "技嘉",
"3017": "奇鋐", "3324": "雙鴻", "2603": "長榮", "2609": "陽明", "2615": "萬海",
"0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益台灣精選高息",
}
=========================================================
#工具函式與資料結構
=========================================================
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
# 第二層：個股風險 (SRI) 基準權重（實際會由動態權重狀態機依 MRI 自動切換）
w_mri_transmission: float = 0.400
w_idiosyncratic: float = 0.600
w_stock_chip_in_idio: float = 0.500
w_stock_health_in_idio: float = 0.250
w_stock_outlook_in_idio: float = 0.250
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
@dataclass
class ChipPainReport:
status: str                          # 多殺多預警 / 籌碼乾淨 / 中性正常 / 暫無融資資料
risk_adj: float                      # 對籌碼風險分的加減分 (-10 ~ +15)
vwap_20: Optional[float] = None
price_vs_vwap_pct: Optional[float] = None
margin_chg_20d_pct: Optional[float] = None
note: str = ""
@dataclass
class DownsideAlphaReport:
has_badge: bool                      # 是否獲得「🛡️ 逆風抗跌徽章」
down_days_count: int = 0
avg_stock_ret_pct: Optional[float] = None
avg_twii_ret_pct: Optional[float] = None
avg_inst_net_lots: Optional[float] = None
note: str = ""
@dataclass
class AdvancedRiskOverlay:
regime_name: str
w_mri_dynamic: float
w_idio_dynamic: float
divergence_penalty: float
divergence_warning: Optional[str]
pain_report: ChipPainReport
downside_alpha: DownsideAlphaReport
=========================================================
#進階量化因子演算法（狀態機 / 背離懲罰 / 痛點指數 / 逆風韌性）
=========================================================
def get_dynamic_regime_weights(mri_score: float) -> Tuple[float, float, str]:
"""
1. 動態權重狀態機 (Regime-Switching)：
- 低波動多頭期 (MRI < 45)：大盤 15% / 個股 85%
- 系統性恐慌期 (MRI >= 70)：大盤 65% / 個股 35%
- 常態震盪期 (45 <= MRI < 70)：大盤 40% / 個股 60%
"""
if mri_score < 45.0:
return 0.15, 0.85, "低波動多頭期 (15/85)"
elif mri_score >= 70.0:
return 0.65, 0.35, "系統性恐慌期 (65/35)"
return 0.40, 0.60, "常態震盪期 (40/60)"
def detect_divergence_penalty(
quality_risk: float,
outlook_risk: float,
chip_risk: float,
trend_risk: float,
) -> Tuple[float, Optional[str]]:
"""
2. 訊號背離非線性陷阱偵測 (Divergence Penalty)：
基本面/展望極好（低風險分）但籌碼/趨勢極差（高風險分）時，觸發額外 +15 ~ +20 懲罰。
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
3. 籌碼痛點指數 (Pain Index)：
結合 20 日成交量加權均價 (VWAP) 與近 20 日融資餘額變化，判斷槓桿套牢多殺多或籌碼沉澱。
"""
if stock_df is None or len(stock_df) < 20:
return ChipPainReport(status="資料不足", risk_adj=0.0, note="交易日不足 20 天")
df20 = stock_df.tail(20).copy()
typical_price = (df20["High"] + df20["Low"] + df20["Close"]) / 3.0
vol_sum = float(df20["Volume"].sum())
vwap_20 = float((typical_price * df20["Volume"]).sum() / vol_sum) if vol_sum > 0 else float(df20["Close"].mean())
latest_close = float(df20["Close"].iloc[-1])
price_vs_vwap_pct = round((latest_close / vwap_20 - 1.0) * 100.0, 2) if vwap_20 > 0 else None
margin_chg_pct = None
if margin_series is not None:
m = pd.to_numeric(margin_series, errors="coerce").dropna()
if len(m) >= 20 and m.iloc[-20] > 0:
margin_chg_pct = round((m.iloc[-1] / m.iloc[-20] - 1.0) * 100.0, 1)
if price_vs_vwap_pct is None or margin_chg_pct is None:
return ChipPainReport(
status="暫無融資資料",
risk_adj=0.0,
vwap_20=round(vwap_20, 2),
price_vs_vwap_pct=price_vs_vwap_pct,
note="無法取得完整 20 日融資餘額（可能為 ETF 或無信用交易）",
)
if margin_chg_pct >= 10.0 and price_vs_vwap_pct <= -8.0:
return ChipPainReport(
status="多殺多預警",
risk_adj=15.0,
vwap_20=round(vwap_20, 2),
price_vs_vwap_pct=price_vs_vwap_pct,
margin_chg_20d_pct=margin_chg_pct,
note=f"近20日融資大增 {margin_chg_pct:+.1f}%，但現價低於20日均價 {price_vs_vwap_pct:.1f}%，留意斷頭賣壓",
)
if margin_chg_pct <= -5.0 and price_vs_vwap_pct >= 0.0:
return ChipPainReport(
status="籌碼乾淨",
risk_adj=-10.0,
vwap_20=round(vwap_20, 2),
price_vs_vwap_pct=price_vs_vwap_pct,
margin_chg_20d_pct=margin_chg_pct,
note=f"近20日融資減少 {margin_chg_pct:+.1f}% 且站穩20日均價 ({price_vs_vwap_pct:+.1f}%)，下檔支撐強",
)
return ChipPainReport(
status="中性正常",
risk_adj=0.0,
vwap_20=round(vwap_20, 2),
price_vs_vwap_pct=price_vs_vwap_pct,
margin_chg_20d_pct=margin_chg_pct,
note=f"現價距20日均價 {price_vs_vwap_pct:+.1f}%，近20日融資變動 {margin_chg_pct:+.1f}%",
)
def compute_downside_alpha(
stock_df: pd.DataFrame,
twii_df: pd.DataFrame,
inst_net_shares_series: Optional[pd.Series] = None,
lookback_days: int = 60,
) -> DownsideAlphaReport:
"""
4. 逆風韌性係數 (Downside Alpha)：
回頭掃描過去 60 個交易日中「加權指數單日跌幅超過 1%」的日子，計算個股平均漲跌幅與法人淨買超。
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
avg_inst_lots = round(float(inst_aligned.mean()) / 1000.0, 1)  # 股數換算張數
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
=========================================================
#後台消息面與資料擷取層（附 Streamlit 雲端快取）
=========================================================
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
clean_id: str, resolved_sym: str, latest_close: float, token: str
) -> Tuple[FundamentalQualityReport, ForwardOutlookReport, str]:
"""
智慧基本面＋未來展望擷取（Yahoo + FinMind 雙重備援）：
- 歷史體質：ROE、營益率、營收年增率、PE、PB
- 未來展望：Forward vs Trailing PE、法人目標價空間、近3月 vs 近12月營收加速度
- 僅在成功抓到真實數據時快取 30 分鐘；若全為空值絕不快取
"""
if "fund_outlook_cache" not in st.session_state:
st.session_state["fund_outlook_cache"] = {}
cache_key = f"{clean_id}_{bool(token)}_v84"
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
# 5) 評分 B：未來展望與法人預期（3 大前瞻因子）
o_score = 50
o_strengths, o_warns = [], []
implied_eps_g = None
target_upside = None
if trailing_pe is not None and forward_pe is not None and trailing_pe > 0 and forward_pe > 0:
implied_eps_g = round((trailing_pe / forward_pe - 1.0) * 100.0, 1)
if implied_eps_g >= 20.0:
o_score += 16
o_strengths.append(f"預估未來一年獲利高成長 ({implied_eps_g:+.1f}%，Forward PE 降至 {forward_pe}x)")
elif implied_eps_g >= 8.0:
o_score += 9
o_strengths.append(f"預估獲利穩步擴張 ({implied_eps_g:+.1f}%)")
elif implied_eps_g <= -10.0:
o_score -= 14
o_warns.append(f"市場預估獲利恐衰退 ({implied_eps_g:+.1f}%，Forward PE 升至 {forward_pe}x)")
elif implied_eps_g <= -3.0:
o_score -= 7
o_warns.append(f"預估獲利微幅下修 ({implied_eps_g:+.1f}%)")
if target_mean is not None and latest_close > 0:
target_upside = round((target_mean / latest_close - 1.0) * 100.0, 1)
cnt_txt = f"({analyst_cnt}家機構)" if analyst_cnt else ""
if target_upside >= 20.0:
o_score += 16
o_strengths.append(f"法人共識目標價 {target_mean:.1f} 元{cnt_txt}，潛在空間達 {target_upside:+.1f}%")
elif target_upside >= 8.0:
o_score += 8
o_strengths.append(f"法人目標價 {target_mean:.1f} 元{cnt_txt}，尚有 {target_upside:+.1f}% 空間")
elif target_upside <= -10.0:
o_score -= 14
o_warns.append(f"現價已明顯高於法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)，留意追高")
elif target_upside <= -3.0:
o_score -= 6
o_warns.append(f"現價逼近法人目標價 {target_mean:.1f} 元 ({target_upside:+.1f}%)")
if rev_accel is not None:
if rev_accel >= 15.0:
o_score += 16
o_strengths.append(f"近 3 月營收動能顯著加速 (較年均 {rev_accel:+.1f}%)")
elif rev_accel >= 5.0:
o_score += 8
o_strengths.append(f"近 3 月營收溫和增溫 (較年均 {rev_accel:+.1f}%)")
elif rev_accel <= -12.0:
o_score -= 14
o_warns.append(f"近 3 月營收動能明顯鈍化 (較年均 {rev_accel:+.1f}%)")
elif rev_accel <= -4.0:
o_score -= 7
o_warns.append(f"近 3 月營收略低於年均 ({rev_accel:+.1f}%)")
has_outlook = any(v is not None for v in (implied_eps_g, target_upside, rev_accel))
if not has_outlook:
o_warns.append("目前缺乏外資預估與月營收趨勢資料，展望採中性 50 分基準")
o_score = int(np.clip(o_score, 10, 95))
o_rating = (
"強勁看多" if o_score >= 75 else
"正向偏多" if o_score >= 60 else
"中性觀望" if o_score >= 45 else
"動能轉弱" if o_score >= 32 else "保守偏空"
)
outlook = ForwardOutlookReport(
outlook_score=o_score, risk_equivalent=float(100 - o_score), rating=o_rating,
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
"""快取個股歷史股價與籌碼資料"""
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
=========================================================
#v9.0 核心風控運算引擎（含動態狀態機、背離懲罰、痛點指數、逆風韌性）
=========================================================
class InstitutionalQuantRiskEngine:
def init(self, config: Optional[RiskModelConfig] = None):
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
s_cmd = float(np.clip(s_cmd, 10, 100))
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
sm_5d, sm_10d = row.get("smart_5d_vol_ratio", np.nan), row.get("smart_10d_vol_ratio", np.nan)
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
"""計算個股技術趨勢風險分（用於訊號背離偵測）"""
close_p = float(row.get("known_stk_close", 0.0))
vwap_20 = float(row.get("stk_vwap_20", close_p))
dd_20 = float(row.get("stk_dd_20d", 0.0))
t_risk = 45.0
if vwap_20 > 0 and close_p < vwap_20:
gap_pct = (close_p / vwap_20 - 1.0) * 100.0
t_risk += min(abs(gap_pct) * 3.0, 28.0)
else:
t_risk -= 10.0
if pd.notna(dd_20) and dd_20 <= -8.0:
t_risk += min(abs(dd_20 + 8.0) * 1.5, 20.0)
return float(np.clip(t_risk, 10.0, 95.0))
def _get_lock_floor(self, row: pd.Series) -> float:
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
"""樂觀化動態目標價與下檔防守價計算（納入體質與未來展望綜合加成）"""
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
latest_close_val = float(stk["Close"].dropna().iloc[-1])
# 智慧基本面＋未來展望查詢（雙重備援，空值不快取）
quality, outlook, yf_name = fetch_fundamentals_and_outlook_smart(
clean_id, resolved_sym, latest_close_val, token
)
name_map = fetch_tw_stock_name_map(token)
stock_name = name_map.get(clean_id) or yf_name
display_name = f"{clean_id}-{stock_name}" if stock_name else clean_id
live_news = fetch_news_risk_score("") if fast_batch else fetch_news_risk_score(stock_id)
mkt_note = "真實籌碼" if not mkt_chips.empty else f"備援({mkt_status})"
stk_note = "真實籌碼" if not stk_chips.empty else f"資金流代理({stk_status})"
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
df["stk_high_15d"] = stk_h.rolling(15).max()
df["stk_vwap_20"] = (stk_c * stk_v).rolling(20).sum() / stk_v.rolling(20).sum().replace(0, np.nan)
hl_range = (stk_h - stk_l).replace(0, np.nan)
mfm = (((stk_c - stk_l) - (stk_h - stk_c)) / hl_range).fillna(0.0)
df["stk_cmf_10"] = (mfm * stk_v).rolling(10).sum() / stk_v.rolling(10).sum().replace(0, np.nan)
df["stk_dd_20d"] = (stk_c / stk_c.rolling(20, min_periods=5).max() - 1) * 100
def align(k: str) -> pd.Series:
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
df["smart_10d_vol_ratio"] = (sc["smart_money_net"].rolling(10).sum() / v10) * 100
is_sell, is_buy = (sc["trust_net"] < 0).astype(int), (sc["trust_net"] > 0).astype(int)
df["trust_consec_sell"] = np.where(is_sell == 1, is_sell.groupby((is_sell != is_sell.shift()).cumsum()).cumsum(), 0)
df["trust_consec_buy"] = np.where(is_buy == 1, is_buy.groupby((is_buy != is_buy.shift()).cumsum()).cumsum(), 0)
df["trust_prev15_net"] = sc["trust_net"].shift(3).rolling(15).sum()
# 預先計算進階因子 3（籌碼痛點指數）與因子 4（逆風韌性係數）
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
latest_overlay = None
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
# 因子 1：動態權重狀態機
w_mri_dyn, w_idio_dyn, regime_name = get_dynamic_regime_weights(mri_score)
# 因子 3：籌碼痛點修正
chip_smooth = float(np.average(c_win, weights=self.chip_weights[-len(c_win):]))
adj_chip = float(np.clip(chip_smooth + pain_report.risk_adj, 5.0, 98.0))
idio = (
adj_chip * self.w.w_stock_chip_in_idio
+ quality.risk_equivalent * self.w.w_stock_health_in_idio
+ outlook.risk_equivalent * self.w.w_stock_outlook_in_idio
)
# 因子 2：訊號背離非線性懲罰
trend_risk = self.score_stock_trend_risk(row)
div_penalty, div_warn = detect_divergence_penalty(
quality_risk=quality.risk_equivalent,
outlook_risk=outlook.risk_equivalent,
chip_risk=adj_chip,
trend_risk=trend_risk,
)
raw_sri = mri_score * w_mri_dyn + idio * w_idio_dyn + div_penalty + resilience_bonus
if adj_chip > 68.0:
raw_sri += min((adj_chip - 68.0) * 0.45, 10.0)
sri_score = int(np.clip(round(raw_sri), 0, 100))
if is_last:
latest_overlay = AdvancedRiskOverlay(
regime_name=regime_name,
w_mri_dynamic=w_mri_dyn,
w_idio_dynamic=w_idio_dyn,
divergence_penalty=div_penalty,
divergence_warning=div_warn,
pain_report=pain_report,
downside_alpha=downside_report,
)
sell_p, stop_p = self._calc_prices(sri_score, mri_score, combined_q, row)
history.append({
"date": work_df.index[i],
"close": round(float(row["known_stk_close"]), 2),
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
=========================================================
#卡片渲染：進階風控偵測、財務體質、未來展望、Plotly 深色圖表
=========================================================
def render_advanced_overlay_card(ov: AdvancedRiskOverlay):
badges_html = f'<span class="badge-yellow">狀態機：{ov.regime_name}</span> '
if ov.downside_alpha.has_badge:
badges_html += '<span class="badge-green">🛡️ 逆風抗跌徽章</span> '
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
st.markdown(
f"""<div class="risk-card">
<div class="card-title">🧠 進階量化風控偵測（動態狀態機 / 籌碼痛點 / 逆風韌性）</div>
<div style="margin:6px 0 8px 0;">{badges_html}</div>
{div_html}
<div class="health-list">
<div>• <b>籌碼痛點 (20日VWAP)：</b>{ov.pain_report.note}</div>
<div>• <b>逆風韌性 (Downside Alpha)：</b>{ov.downside_alpha.note}</div>
</div>
</div>""",
unsafe_allow_html=True,
)
def _metric_tag(val, good_cond, bad_cond, good_txt, mid_txt, bad_txt, empty_txt="無資料"):
if val is None or pd.isna(val):
return f'<span class="m-tag" style="color:#6e7681;">{empty_txt}</span>'
if good_cond(val):
return f'<span class="m-tag tag-good">{good_txt}</span>'
if bad_cond(val):
return f'<span class="m-tag tag-bad">{bad_txt}</span>'
return f'<span class="m-tag tag-mid">{mid_txt}</span>'
def render_health_card(q: FundamentalQualityReport):
roe_t = _metric_tag(q.roe_pct, lambda x: x >= 15, lambda x: x < 5, "優異", "穩健", "偏弱")
op_t = _metric_tag(q.op_margin_pct, lambda x: x >= 15, lambda x: x < 3, "高毛利", "普通", "偏薄")
rev_t = _metric_tag(q.rev_growth_pct, lambda x: x >= 15, lambda x: x <= -10, "高成長", "平穩", "衰退")
pe_t = _metric_tag(q.pe, lambda x: 0 < x <= 16, lambda x: x >= 38 or x < 0, "便宜", "合理", "偏貴/虧損")
pb_t = _metric_tag(q.pb, lambda x: 0 < x <= 2.0, lambda x: x >= 6.0, "低估", "合理", "溢價")
pb_str = f"{q.pb:.2f}x" if q.pb is not None else "N/A"
str_html = "".join([f"<div>✅ {s}</div>" for s in q.strengths])
warn_html = "".join([f"<div>⚠️ {w}</div>" for w in q.warnings]) if q.warnings else "<div>✨ 目前無明顯財務警訊</div>"
st.markdown(
f"""<div class="risk-card">
<div class="card-title">📊 個股財務體質健檢（來源：{q.source_note}）</div>
<div style="font-size:1.35rem;font-weight:700;color:#f0f6fc;">{q.quality_score} <span style="font-size:0.95rem;color:#8b949e;">/ 100</span>  【{q.rating}】</div>
<div class="metric-grid">
<div class="metric-box"><div class="m-label">ROE 報酬率</div><div class="m-val">{_fmt_num(q.roe_pct, '%')}</div>{roe_t}</div>
<div class="metric-box"><div class="m-label">營業利益率</div><div class="m-val">{_fmt_num(q.op_margin_pct, '%')}</div>{op_t}</div>
<div class="metric-box"><div class="m-label">營收年增率</div><div class="m-val">{_fmt_num(q.rev_growth_pct, '%', signed=True)}</div>{rev_t}</div>
<div class="metric-box"><div class="m-label">本益比 PE</div><div class="m-val">{_fmt_num(q.pe, 'x')}</div>{pe_t}</div>
<div class="metric-box"><div class="m-label">淨值比 PB</div><div class="m-val">{pb_str}</div>{pb_t}</div>
</div>
<div class="health-list">{str_html}{warn_html}</div>
</div>""",
unsafe_allow_html=True,
)
def render_outlook_card(o: ForwardOutlookReport):
eps_t = _metric_tag(o.implied_eps_growth_pct, lambda x: x >= 10, lambda x: x <= -5, "看增", "平穩", "看減", "無預估")
tgt_t = _metric_tag(o.target_upside_pct, lambda x: x >= 10, lambda x: x <= -5, "具空間", "合理", "偏貴", "無外資追蹤")
acc_t = _metric_tag(o.rev_accel_pct, lambda x: x >= 5, lambda x: x <= -5, "加速中", "持平", "鈍化", "無資料")
pe_sub = f"預估 {o.forward_pe}x / 過去 {o.trailing_pe}x" if (o.forward_pe and o.trailing_pe) else "需雙 PE 數據"
tgt_sub = f"目標 {o.target_mean_price:.1f} 元" if o.target_mean_price else "中小型股常見"
acc_sub = "近 3 月 vs 近 12 月"
score_col = "#3fb950" if o.outlook_score >= 60 else ("#ff7b72" if o.outlook_score < 45 else "#e3b341")
str_html = "".join([f"<div>🚀 {s}</div>" for s in o.strengths])
warn_html = "".join([f"<div>⚠️ {w}</div>" for w in o.warnings]) if o.warnings else ""
st.markdown(
f"""<div class="risk-card">
<div class="card-title">🔮 未來展望與法人預期</div>
<div style="font-size:1.35rem;font-weight:700;color:{score_col};">{o.outlook_score} <span style="font-size:0.95rem;color:#8b949e;">/ 100</span>  【{o.rating}】</div>
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
</div>""",
unsafe_allow_html=True,
)
def render_dark_chart(res: Dict):
df = res["hist_df"]
fig = make_subplots(
rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.62, 0.38]
)
fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="收盤價", line=dict(color="#58a6ff", width=2.5)), row=1, col=1)
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
=========================================================
#側邊欄：FinMind Token 設定（選填）
=========================================================
with st.sidebar:
st.markdown("### ⚙️ 系統連線設定")
st.text_input(
"FinMind API Token（選填）",
type="password",
key="user_finmind_token",
help="若遇到免費限流，可至 finmindtrade.com 免費申請 Token 填入。",
)
st.caption(
"燈號標準：\n🟢 安全 (<48)\n🟡 中等 (48~64)\n🔴 危險 (≥65)\n\n"
"動態狀態機權重：\n"
"• MRI < 45：大盤 15% / 個股 85%\n"
"• 45 ≤ MRI < 70：大盤 40% / 個股 60%\n"
"• MRI ≥ 70：大盤 65% / 個股 35%"
)
=========================================================
#主介面：雙分頁（單股一鍵查閱 / 多股批量掃描）
=========================================================
st.markdown("## 🛡️ 台股量化風控儀表板")
tab1, tab2 = st.tabs(["🔍 單股風控診斷", "📋 多股批量掃描"])
engine = InstitutionalQuantRiskEngine()
with tab1:
col_in, col_btn = st.columns([3, 1.2])
with col_in:
stock_input = st.text_input("輸入台股代號（上市／上櫃自動偵測）", value="2330", placeholder="例如 2330、2382、6274", label_visibility="collapsed")
with col_btn:
run_btn = st.button("一鍵分析", type="primary", use_container_width=True)
if stock_input.strip():
with st.spinner(f"正在運算 {stock_input.strip()} 量化風控、體質與前瞻模型..."):
try:
res = engine.run_analysis(stock_input.strip())
m_icon, m_stat, m_col, m_cls = get_traffic_light(res["mri"])
s_icon, s_stat, s_col, s_cls = get_traffic_light(res["sri"])
sell_pct = (res["sell_p"] / res["close"] - 1) * 100
stop_pct = (res["stop_p"] / res["close"] - 1) * 100
ov: AdvancedRiskOverlay = res["overlay"]
mri_weight_pct = int(round(ov.w_mri_dynamic * 100))
st.markdown(f"### {res['display_name']}")
st.caption(f"基準日：{res['date']} ｜ 收盤價：{res['close']} 元 ｜ 籌碼：{res['stk_note']}")
# 第一排：個股風險 vs 大盤風險
c1, c2 = st.columns(2)
with c1:
st.markdown(
f"""<div class="risk-card">
<div class="card-title">個股風險指數 (SRI)</div>
<div class="card-value" style="color:{s_col};">{s_icon} {res['sri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
<div class="card-sub"><span class="{s_cls}">{s_stat}</span>  籌碼風險 {res['chip']}/100</div>
</div>""",
unsafe_allow_html=True,
)
with c2:
st.markdown(
f"""<div class="risk-card">
<div class="card-title">大盤風險指數 (MRI)</div>
<div class="card-value" style="color:{m_col};">{m_icon} {res['mri']} <span style="font-size:1rem;color:#8b949e;">/ 100</span></div>
<div class="card-sub"><span class="{m_cls}">{m_stat}</span>  動態權重 {mri_weight_pct}%</div>
</div>""",
unsafe_allow_html=True,
)
# 第二排：動態價位卡
c3, c4 = st.columns(2)
with c3:
st.markdown(
f"""<div class="risk-card">
<div class="card-title">建議賣出價（目標調節區）</div>
<div class="card-value" style="color:#d2a8ff;">{res['sell_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
<div class="card-sub">距收盤 {sell_pct:+.1f}%</div>
</div>""",
unsafe_allow_html=True,
)
with c4:
st.markdown(
f"""<div class="risk-card">
<div class="card-title">下檔防守價</div>
<div class="card-value" style="color:#f0883e;">{res['stop_p']:.1f} <span style="font-size:0.9rem;">元</span></div>
<div class="card-sub">距收盤 {stop_pct:+.1f}%</div>
</div>""",
unsafe_allow_html=True,
)
# 進階量化風控偵測卡片（動態狀態機、背離懲罰、籌碼痛點、逆風韌性）
render_advanced_overlay_card(ov)
# 第三排：未來展望與法人預期獨立卡片
render_outlook_card(res["outlook"])
# 第四排：歷史財務體質健檢卡
render_health_card(res["quality"])
# 第五排：互動式走勢圖
render_dark_chart(res)
except Exception as e:
st.error(f"分析失敗：{e}")
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
tags = []
if ov_b.downside_alpha.has_badge:
tags.append("🛡️抗跌")
if ov_b.divergence_warning:
tags.append("⚠️利多背離")
if ov_b.pain_report.status == "多殺多預警":
tags.append("🔥多殺多")
elif ov_b.pain_report.status == "籌碼乾淨":
tags.append("💎乾淨")
rows.append({
"標的": r["display_name"],
"收盤價": r["close"],
"個股風險": f"{s_i} {r['sri']} ({s_st})",
"進階訊號": " ".join(tags) if tags else "-",
"籌碼分": r["chip"],
"未來展望": f"{r['outlook'].outlook_score} ({r['outlook'].rating})",
"體質評等": f"{r['quality'].quality_score} ({r['quality'].rating.split()[0]})",
"建議賣出價": f"{r['sell_p']:.1f} ({sell_pct:+.1f}%)",
"下檔防守價": r["stop_p"],
"大盤MRI": f"{m_i} {r['mri']}",
})
except Exception as e:
rows.append({
"標的": tid,
"收盤價": None,
"個股風險": f"失敗({str(e)[:12]})",
"進階訊號": "-",
"籌碼分": None,
"未來展望": "-",
"體質評等": "-",
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
st.markdown(
'<div class="disclaimer">本系統所有數據與風險指標皆由公開資訊與量化演算法自動運算產生，僅供風險控管參考，絕非投資買賣建議。
投資人應獨立判斷並自負盈虧與法律責任。</div>',
unsafe_allow_html=True,
)
