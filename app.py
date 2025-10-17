# App.py — Panel de Finanzas con Google Sheets (Transactions) + Gemini + Plotly
# Incluye:
# - Optimización LIBRE por tickers (sección superior)
# - Debajo: expander "Realizar análisis con mi portafolio" con pestañas "Equity-only" y "ETFs-only"

import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== Google Sheets (Service Account) =====
from google.oauth2.service_account import Credentials
import gspread

# =========================================================
# CONFIG GENERAL
# =========================================================
st.set_page_config(page_title="Finanzas – Panel", page_icon="📈", layout="wide")

# Estilo sobrio/profesional
st.markdown("""
<style>
.main .block-container {max-width: 1220px; padding-top: 1rem; padding-bottom: 2rem;}
h1,h2,h3 {font-weight:700; letter-spacing:-.2px}
.section-subtitle { color:#6b7280; margin-bottom: 1.0rem; }
.hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.0rem 0; }
.stat-label { color:#6b7280; font-size:.88rem; margin-bottom:.2rem }
.stat-value { font-weight:700; font-size:1.1rem }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; height:92px; }
.stSidebar { border-right:1px solid #e5e7eb; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:.9rem; }
.badge-green { background:#e7f8ec; color:#166534; border:1px solid #bbf7d0; }
.badge-yellow{ background:#fff7e6; color:#92400e; border:1px solid #fde68a; }
.badge-red   { background:#fee2e2; color:#7f1d1d; border:1px solid #fecaca; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("### Menú")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Consulta de Acciones",
        "Mi Portafolio (Google Sheets)",
        "Evaluar nueva acción (básico)",
        "Riesgo de Inversión",
        "CAPM",
        "Optimización de Portafolio (Markowitz)",
        "Simulación Monte Carlo",
    ],
)

# =========================================================
# Periodos estándar
# =========================================================
RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "1 año", "YTD", "3 años", "5 años"]

def range_to_yf_params(range_key: str):
    mapping = {
        "1 semana": ("7d", "1d"),
        "1 mes": ("1mo", "1d"),
        "6 meses": ("6mo", "1d"),
        "1 año": ("1y", "1d"),
        "YTD": ("ytd", "1d"),
        "3 años": ("3y", "1wk"),
        "5 años": ("5y", "1wk"),
    }
    return mapping.get(range_key, ("6mo", "1d"))

# =========================================================
# HELPERS: YFINANCE
# =========================================================
@st.cache_data(ttl=3600)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False,
                             progress=False, threads=False)
    except Exception:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False,
                         progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    required = {"Open","High","Low","Close","Volume"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df.dropna(subset=["Open","High","Low","Close"])

@st.cache_data(ttl=1200)
def get_last_close(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    try:
        df = yf.download(symbol, period="5d", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return np.nan

@st.cache_data(ttl=1800)
def load_prices(tickers, period="5y", interval="1d") -> pd.DataFrame:
    if isinstance(tickers, str): tickers = [tickers]
    tickers = [t for t in tickers if t]
    if not tickers: return pd.DataFrame()
    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        else:
            first = raw.columns.levels[0][0]
            prices = raw[first].copy()
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy().rename(columns={"Adj Close":"Close"})
        else:
            return pd.DataFrame()
        prices.columns = [tickers[0]]
    return prices.dropna(how="all")

def return_metrics(df: pd.DataFrame):
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["LogReturn"] = np.log(d["Close"] / d["Close"].shift(1))
    d = d.dropna()
    avg_daily = d["Return"].mean()
    ann_return = (1 + avg_daily) ** 252 - 1
    ann_vol = d["Return"].std() * np.sqrt(252)
    return d, ann_return, ann_vol

def compute_beta_alpha(stock_returns: pd.Series, market_returns: pd.Series):
    merged = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if merged.shape[0] < 2: return np.nan, np.nan
    cov = np.cov(merged.iloc[:,0], merged.iloc[:,1])[0][1]
    var_m = merged.iloc[:,1].var()
    beta = cov/var_m if var_m != 0 else np.nan
    alpha = merged.iloc[:,0].mean() - beta*merged.iloc[:,1].mean()
    return beta, alpha

def max_drawdown(series: pd.Series):
    if series.empty: return np.nan
    cummax = series.cummax()
    dd = (series / cummax) - 1.0
    return dd.min()

# =========================================================
# FORMATEADORES
# =========================================================
def human_number(n):
    try: n = float(n)
    except Exception: return "N/A"
    absn = abs(n)
    if absn >= 1e12: return f"{n/1e12:.2f}T"
    if absn >= 1e9:  return f"{n/1e9:.2f}B"
    if absn >= 1e6:  return f"{n/1e6:.2f}M"
    if absn >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

def pct(x):
    try: return f"{float(x)*100:.2f}%"
    except Exception: return "N/A"

def ratio(x):
    try: return f"{float(x):.2f}"
    except Exception: return "N/A"

# =========================================================
# GRÁFICOS (Plotly)
# =========================================================
TEMPLATE = "simple_white"
COLOR_UP, COLOR_UP_FILL = "rgba(22,163,74,1)", "rgba(22,163,74,0.9)"
COLOR_DOWN, COLOR_DOWN_FILL = "rgba(220,38,38,1)", "rgba(220,38,38,0.9)"
SMA_COLORS = {"SMA20":"#5546d6","SMA50":"#f59e0b","SMA200":"#b78d0a"}

def add_smas(df: pd.DataFrame, windows=(20,50,200)):
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out

def plot_candles_with_volume(df: pd.DataFrame, title: str,
                             show_sma20=True, show_sma50=True, show_sma200=False):
    df = df.copy(); df.index = pd.to_datetime(df.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.72,0.28])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color=COLOR_UP, increasing_fillcolor=COLOR_UP_FILL,
        decreasing_line_color=COLOR_DOWN, decreasing_fillcolor=COLOR_DOWN_FILL,
        line=dict(width=1.2), whiskerwidth=0.5, opacity=0.95
    ), row=1, col=1)

    for k,flag in [("SMA20",show_sma20),("SMA50",show_sma50),("SMA200",show_sma200)]:
        if flag and k in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[k], mode="lines",
                                     line=dict(color=SMA_COLORS[k], width=1.6),
                                     name=k), row=1, col=1)
    vol_colors = [COLOR_UP_FILL if c>=o else COLOR_DOWN_FILL for o,c in zip(df["Open"],df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors,
                         name="Volumen", opacity=0.85), row=2, col=1)

    fig.update_xaxes(
        showgrid=False,
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="year", stepmode="todate", label="YTD"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        rangebreaks=[dict(bounds=["sat","mon"])],
        row=1, col=1
    )
    fig.update_xaxes(showgrid=False, rangebreaks=[dict(bounds=["sat","mon"])], row=2, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Volumen", showgrid=True, zeroline=False, row=2, col=1)
    fig.update_layout(title=title, template=TEMPLATE, height=760, margin=dict(l=40,r=20,t=48,b=30),
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# =========================================================
# GEMINI (traducción)
# =========================================================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
BASES = ["https://generativelanguage.googleapis.com/v1","https://generativelanguage.googleapis.com/v1beta"]
PREFERRED = ["gemini-1.5-flash-latest","gemini-1.5-flash","gemini-1.5-flash-8b","gemini-1.0-pro","gemini-pro"]

def list_models(base: str):
    try:
        r = requests.get(f"{base}/models", params={"key": API_KEY}, timeout=20)
        if r.status_code == 200: return True, r.json()
        return False, r.text
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=3600)
def pick_model_and_base():
    if not API_KEY: return None, None
    last_err = None
    for base in BASES:
        ok, data = list_models(base)
        if not ok:
            last_err = data; continue
        models = data.get("models", []) if isinstance(data, dict) else []
        supported = set()
        for m in models:
            name = m.get("name")
            methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
            if not name: continue
            short = name.split("/")[-1]
            if any(x in ("generateContent","generate_content") for x in methods):
                supported.add(short)
        for pref in PREFERRED:
            if pref in supported: return base, pref
        if supported: return base, sorted(supported)[0]
        last_err = "No hay modelos con generateContent."
    return None, None

def generate_content_rest(base: str, model: str, text: str) -> str:
    url = f"{base}/models/{model}:generateContent"
    payload = {"contents":[{"parts":[{"text": text}]}]}
    r = requests.post(url, params={"key": API_KEY},
                      headers={"Content-Type":"application/json"},
                      data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        try: err = r.json()
        except Exception: err = r.text
        raise RuntimeError(f"{r.status_code} {err}")
    data = r.json()
    for cand in data.get("candidates", []):
        content = cand.get("content") or {}
        for p in content.get("parts", []):
            txt = (p.get("text") or "").strip()
            if txt: return txt
    return ""

def translate_to_spanish(text: str) -> str:
    if not text or not API_KEY: return ""
    base, model = pick_model_and_base()
    if not base or not model:
        raise RuntimeError("No se pudo seleccionar modelo/base de Gemini. Revisa GOOGLE_API_KEY.")
    prompt = ("Traduce al español el siguiente texto sobre una empresa. "
              "Usa español claro y natural de negocios. Conserva nombres propios y números. "
              "Solo devuelve la traducción, sin explicaciones.\n\n" + text)
    return generate_content_rest(base, model, prompt)

# =========================================================
# GOOGLE SHEETS — conexión y utilidades (Worksheet NO cacheado)
# =========================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_gs_client():
    info = st.secrets.get("gcp_service_account", None)
    creds = None
    if info:
        creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
    elif os.path.exists("credentials.json"):
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    else:
        return None
    return gspread.authorize(creds)

def open_or_create_transactions_ws():
    client = get_gs_client()
    if client is None:
        return None
    key = st.secrets.get("GSHEETS_SPREADSHEET_NAME", "").strip()
    ws_name = st.secrets.get("GSHEETS_TRANSACTIONS_WS", "Transactions").strip() or "Transactions"
    if not key: return None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9\-_]+)", key)
    if m: key = m.group(1)
    sh = client.open_by_key(key)
    try:
        ws = sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=1000, cols=10)
        ws.update([["Ticker","TradeDate","Shares","Price","Fees","Notes"]])
    vals = ws.get_all_values()
    if not vals or [c.strip() for c in vals[0]] != ["Ticker","TradeDate","Shares","Price","Fees","Notes"]:
        ws.clear()
        ws.update([["Ticker","TradeDate","Shares","Price","Fees","Notes"]])
    return ws

def open_transactions_ws(force_retry: bool = True):
    try:
        return open_or_create_transactions_ws()
    except Exception:
        if force_retry:
            st.cache_resource.clear()
            try:
                return open_or_create_transactions_ws()
            except Exception:
                return None
        return None

def read_transactions_df() -> pd.DataFrame:
    ws = open_transactions_ws(force_retry=True)
    if ws is None:
        return pd.DataFrame()
    try:
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header_row_idx = None
        for i, row in enumerate(values):
            row_norm = [str(c).strip() for c in row]
            if "Ticker" in row_norm and "Shares" in row_norm and "TradeDate" in row_norm:
                header_row_idx = i
                headers = row_norm
                break
        if header_row_idx is None:
            headers = [str(c).strip() for c in values[0]]
            data_rows = values[1:]
        else:
            data_rows = values[header_row_idx + 1 :]
        df = pd.DataFrame(data_rows, columns=headers).replace({"": None})
        for col in ["Ticker","TradeDate","Shares","Price","Fees","Notes"]:
            if col not in df.columns:
                df[col] = None
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce")
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Fees"] = pd.to_numeric(df["Fees"], errors="coerce")
        df = df.dropna(subset=["Ticker","Shares"])
        df = df[df["Ticker"] != ""]
        df = df[df["Shares"] > 0]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def append_transaction_row(ticker: str, trade_date: str, shares: float, price: float=None, fees: float=0.0, notes: str=""):
    ws = open_transactions_ws(force_retry=True)
    if ws is None:
        raise RuntimeError("No se pudo abrir la hoja de transacciones. Verifica secrets y permisos.")
    def norm_num(x):
        if x is None: return ""
        try:
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return ""
            return float(x)
        except Exception:
            return ""
    row = [
        ticker.upper().strip(),
        trade_date,
        norm_num(shares),
        norm_num(price),
        norm_num(fees),
        notes or ""
    ]
    values = ws.get_all_values()
    headers = ["Ticker","TradeDate","Shares","Price","Fees","Notes"]
    if not values:
        ws.update([headers, row])
    else:
        ws.append_row(row, value_input_option="USER_ENTERED")

def overwrite_transactions_df(df: pd.DataFrame):
    ws = open_transactions_ws(force_retry=True)
    if ws is None:
        raise RuntimeError("No se pudo abrir la hoja de transacciones.")
    cols = ["Ticker", "TradeDate", "Shares", "Price", "Fees", "Notes"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["TradeDate"] = out["TradeDate"].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp)
        else ("" if pd.isna(x) else str(x))
    )
    for c in ["Shares","Price","Fees"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.nan, np.inf, -np.inf], "")
    ws.clear()
    ws.update([cols] + out.astype(object).values.tolist(), value_input_option="USER_ENTERED")

def delete_ticker_all_rows(ticker: str):
    df = read_transactions_df()
    if df.empty:
        return False
    ticker = ticker.upper().strip()
    filtered = df[df["Ticker"] != ticker].copy()
    if len(filtered) == len(df):
        return False
    overwrite_transactions_df(filtered)
    return True

# =========================================================
# Clasificación simple
# =========================================================
TECH_HIGH = {"NVDA","GOOG","GOOGL","META","MSFT","AMZN","TSLA","AMD","AVGO"}
LOW_VOL = {"KO","PG","JNJ","PEP","COST","WMT","MCD","MRK","HD"}
INDEX_ETF = {"SPY","VOO","IVV","^GSPC","QQQ","IWM"}

def classify_ticker(t: str) -> str:
    t = t.upper()
    if t in INDEX_ETF: return "Índice/ETF"
    if t in TECH_HIGH: return "Tecnología (alto beta)"
    if t in LOW_VOL:   return "Defensivo (bajo beta)"
    return "Otros"

# =========================================================
# UTILIDADES DE PORTAFOLIO / OPTIMIZACIÓN
# =========================================================
def prices_for_universe(universe: list, period="3y", interval="1d") -> pd.DataFrame:
    if not universe: return pd.DataFrame()
    return load_prices(universe, period=period, interval=interval)

def returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all").dropna(axis=0, how="any")

def shrink_cov(cov: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Shrinkage simple hacia la diagonal: Σ_shrink = (1-α)Σ + α·diag(Σ)."""
    if cov.empty: return cov
    diag = np.diag(np.diag(cov.values))
    shrunk = (1 - alpha) * cov.values + alpha * diag
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)

def simulate_portfolios(mean_ret: pd.Series,
                        cov: pd.DataFrame,
                        n_sims: int,
                        per_asset_min: float,
                        per_asset_max: float,
                        group_limits: dict | None,
                        classes: dict):
    """
    Simula portafolios aleatorios (Dirichlet + rechazo) cumpliendo:
    - Long-only, sum(w)=1
    - Límites por activo [min,max]
    - Límites por grupo (dict: {"Tecnología (alto beta)": (min,max), ...})
    """
    assets = list(mean_ret.index)
    A = len(assets)
    results = []
    weights_list = []
    tries = 0
    max_tries = n_sims * 30  # margen amplio de reintentos

    def valid_groups(w):
        if not group_limits: return True
        gsum = {}
        for i, a in enumerate(assets):
            g = classes.get(a, "Otros")
            gsum[g] = gsum.get(g, 0.0) + w[i]
        for g,(gmin,gmax) in group_limits.items():
            if gsum.get(g, 0.0) < gmin - 1e-9 or gsum.get(g, 0.0) > gmax + 1e-9:
                return False
        return True

    rng = np.random.default_rng(42)
    while len(weights_list) < n_sims and tries < max_tries:
        tries += 1
        w = rng.random(A); w = w / w.sum()
        if np.any(w < per_asset_min - 1e-9) or np.any(w > per_asset_max + 1e-9):
            continue
        if not valid_groups(w):
            continue
        port_ret = float(np.dot(w, mean_ret.values))
        port_vol = float(np.sqrt(w.T @ cov.values @ w))
        sharpe = port_ret / port_vol if port_vol > 0 else np.nan  # rf se aplica fuera si se quiere
        results.append((port_vol, port_ret, sharpe))
        weights_list.append(w)

    if not weights_list:
        return pd.DataFrame(), []
    res = pd.DataFrame(results, columns=["vol","ret","sharpe"])
    return res, weights_list

def current_weights_from_tx(tx: pd.DataFrame, universe: list) -> pd.Series:
    if tx.empty or not universe: return pd.Series(dtype=float)
    tickers = sorted(set(universe))
    last_prices = {t: get_last_close(t) for t in tickers}
    sub = tx[tx["Ticker"].isin(tickers)].copy()
    if sub.empty: return pd.Series(dtype=float)
    sub["LastPrice"] = sub["Ticker"].map(last_prices)
    sub["MktValue"] = sub["Shares"] * sub["LastPrice"]
    pos = sub.groupby("Ticker", as_index=False)["MktValue"].sum()
    total = pos["MktValue"].sum()
    if total <= 0: return pd.Series(dtype=float)
    w = pos.set_index("Ticker")["MktValue"] / total
    for t in tickers:
        if t not in w.index:
            w.loc[t] = 0.0
    return w.sort_index()

def compare_weights_table(current_w: pd.Series, proposed_w: pd.Series) -> pd.DataFrame:
    idx = sorted(set(current_w.index) | set(proposed_w.index))
    out = pd.DataFrame(index=idx, columns=["Actual","Propuesto","Δ (pp)"])
    out["Actual"] = current_w.reindex(idx).fillna(0.0)
    out["Propuesto"] = proposed_w.reindex(idx).fillna(0.0)
    out["Δ (pp)"] = (out["Propuesto"] - out["Actual"]) * 100
    return (out * 100).round(2)

def turnover_abs(current_w: pd.Series, proposed_w: pd.Series) -> float:
    a = current_w.reindex(proposed_w.index).fillna(0.0)
    b = proposed_w.reindex(current_w.index).fillna(0.0)
    return float(np.abs(a - b).sum() / 2.0)

# =========================================================
# SECCIONES
# =========================================================
if menu == "Consulta de Acciones":
    st.markdown("<h1>Consulta de Acciones</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Descripción, indicadores y gráfica interactiva de velas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c0, c1, c2, c3, c4 = st.columns([1.2,1,1,1,1.2])
    with c0:
        stonk = st.text_input("Símbolo", "MSFT").strip().upper()
    with c1:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("6 meses"))
    with c2:
        show_sma20 = st.checkbox("SMA20", value=True)
    with c3:
        show_sma50 = st.checkbox("SMA50", value=True)
    with c4:
        show_sma200 = st.checkbox("SMA200", value=False)

    ticker = yf.Ticker(stonk)
    info = ticker.info if hasattr(ticker, "info") else {}

    st.subheader("Empresa")
    st.write(info.get("longName", "No disponible"))

    st.subheader("Descripción (inglés)")
    summary = info.get("longBusinessSummary", "No disponible.")
    st.write(summary if summary else "No disponible.")

    if st.button("Traducir a español"):
        if not API_KEY:
            st.warning("Configura GOOGLE_API_KEY en Secrets para traducir con Gemini.")
        elif not summary:
            st.warning("No hay descripción para traducir.")
        else:
            with st.spinner("Traduciendo con Gemini..."):
                try:
                    translated = translate_to_spanish(summary)
                    if translated: st.session_state[f"translated_{stonk}"] = translated
                    else: st.warning("No se recibió texto traducido.")
                except Exception as e:
                    st.error(f"Error al traducir con Gemini: {e}")

    if st.session_state.get(f"translated_{stonk}"):
        st.subheader("Descripción (español)")
        st.write(st.session_state[f"translated_{stonk}"])

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Indicadores clave")

    key_data = {
        "Sector": info.get("sector", "N/A"),
        "Industria": info.get("industry", "N/A"),
        "País": info.get("country", "N/A"),
        "Market Cap": human_number(info.get("marketCap")),
        "Beta (5y mensual)": ratio(info.get("beta")),
        "P/E (Trailing)": ratio(info.get("trailingPE")),
        "P/E (Forward)": ratio(info.get("forwardPE")),
        "P/B": ratio(info.get("priceToBook")),
        "Dividend Yield": pct(info.get("dividendYield")),
        "Cambio 52 semanas": pct(info.get("52WeekChange")),
    }
    cols_top = st.columns(5); cols_bottom = st.columns(5)
    keys = list(key_data.keys())
    for i,col in enumerate(cols_top):
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{keys[i]}</div>'
                        f'<div class="stat-value">{key_data[keys[i]]}</div></div>', unsafe_allow_html=True)
    for i,col in enumerate(cols_bottom, start=5):
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{keys[i]}</div>'
                        f'<div class="stat-value">{key_data[keys[i]]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Gráfica de velas con volumen")

    period, interval = range_to_yf_params(range_key)
    hist = get_history(stonk, period, interval)
    if hist.empty:
        st.warning("No se pudo obtener información histórica suficiente. Prueba con otro rango o símbolo.")
    else:
        hist = add_smas(hist, (20,50,200))
        st.plotly_chart(
            plot_candles_with_volume(hist, f"{stonk} · {range_key}",
                                     show_sma20=show_sma20, show_sma50=show_sma50, show_sma200=show_sma200),
            use_container_width=True
        )

elif menu == "Mi Portafolio (Google Sheets)":
    st.markdown("<h1>Mi Portafolio</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Lectura de transacciones, cálculo de pesos por valor de mercado, altas y bajas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if st.button("Reintentar conexión a Sheets"):
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            ws = open_or_create_transactions_ws()
            if ws is not None:
                st.success("Conexión y pestaña Transactions OK.")
                st.rerun()
        except Exception as e:
            st.error(f"Fallo al reconectar: {e}")

    st.subheader("Agregar compra")
    with st.form("add_trade"):
        c1,c2,c3 = st.columns([1.2,1,1])
        with c1: at_ticker = st.text_input("Ticker", "SPY").upper().strip()
        with c2: at_date = st.date_input("Fecha de compra", value=pd.to_datetime("2020-10-16"))
        with c3: at_shares = st.number_input("Acciones", min_value=0.0, value=10.0, step=1.0)
        c4,c5,c6 = st.columns([1,1,2])
        with c4: at_price = st.number_input("Precio de compra (opcional)", min_value=0.0, value=0.0, step=0.01)
        with c5: at_fees = st.number_input("Comisiones (opcional)", min_value=0.0, value=0.0, step=0.01)
        with c6: at_notes = st.text_input("Notas", "Alta manual desde app")
        submitted = st.form_submit_button("Guardar compra")
    if submitted:
        try:
            price_val = None if at_price == 0 else float(at_price)
            append_transaction_row(
                ticker=at_ticker,
                trade_date=pd.to_datetime(at_date).strftime("%Y-%m-%d"),
                shares=float(at_shares),
                price=price_val,
                fees=float(at_fees),
                notes=at_notes,
            )
            st.success(f"Compra de {at_ticker} guardada en Transactions.")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo guardar la compra: {e}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    tx = read_transactions_df()
    if tx.empty:
        st.warning("No se pudieron leer transacciones desde Google Sheets. Verifica secrets/permisos/pestaña 'Transactions'.")
        st.stop()

    tickers = sorted(tx["Ticker"].unique().tolist())
    prices = load_prices(tickers, period="5y", interval="1d")
    if prices.empty:
        st.error("No se pudieron descargar precios para construir el histórico a 5 años.")
        st.stop()

    last_prices = {t: get_last_close(t) for t in tickers}
    tx["LastPrice"] = tx["Ticker"].map(last_prices)
    tx["MktValue"] = tx["Shares"] * tx["LastPrice"]
    pos = tx.groupby("Ticker", as_index=False).agg({"Shares":"sum","MktValue":"sum"})
    total_mv = pos["MktValue"].sum()
    pos["Weight"] = pos["MktValue"] / total_mv if total_mv > 0 else 0.0
    pos["Grupo"] = pos["Ticker"].apply(classify_ticker)

    shares_map = pos.set_index("Ticker")["Shares"].to_dict()
    aligned = prices.copy().reindex(columns=[c for c in prices.columns if c in shares_map], fill_value=np.nan)
    for t in aligned.columns:
        aligned[t] = aligned[t] * shares_map.get(t, 0.0)
    port_value = aligned.sum(axis=1).dropna()
    if port_value.empty:
        st.error("No se pudo construir la serie de valor del portafolio (revisa precios/acciones).")
        st.stop()
    port_ret = port_value.pct_change().dropna()
    ann_ret = (1 + port_ret.mean())**252 - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    rf = st.number_input("Tasa libre de riesgo (Sharpe)", value=0.04, step=0.01, key="rf_pf")
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    mdd = max_drawdown((1+port_ret).cumprod())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rendimiento anualizado", f"{ann_ret*100:.2f}%")
    c2.metric("Volatilidad anualizada", f"{ann_vol*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}")
    c4.metric("Max Drawdown (5y)", f"{mdd*100:.2f}%")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    port_cum = (1 + port_ret).cumprod()
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode="lines", name="Portafolio (valor)"))
    fig_perf.update_layout(title="Crecimiento de $1 · 5 años (basado en tus Shares)", template="simple_white",
                           xaxis_title="", yaxis_title="Crecimiento", height=420)
    st.plotly_chart(fig_perf, use_container_width=True)

    grp = pos.groupby("Grupo", as_index=False)["MktValue"].sum().sort_values("MktValue", ascending=False)
    if not grp.empty:
        fig_grp = go.Figure(data=[go.Pie(labels=grp["Grupo"], values=grp["MktValue"], hole=0.5)])
        fig_grp.update_layout(title="Distribución por grupo (valor de mercado)", template="simple_white", height=380)
        st.plotly_chart(fig_grp, use_container_width=True)

    st.subheader("Detalle por activo")
    pos_view = pos.copy()
    pos_view["LastPrice"] = pos_view["Ticker"].map(last_prices)
    pos_view = pos_view[["Ticker","Grupo","Shares","LastPrice","MktValue","Weight"]]
    pos_view = pos_view.sort_values("MktValue", ascending=False)
    st.dataframe(
        pos_view.style.format({
            "Shares":"{:.2f}",
            "LastPrice":"${:,.2f}",
            "MktValue":"${:,.2f}",
            "Weight":"{:.2%}"
        }),
        use_container_width=True
    )

    st.markdown("##### Eliminar activo del portafolio")
    del_col1, del_col2 = st.columns([1.5,1])
    with del_col1:
        del_ticker = st.selectbox("Selecciona ticker a eliminar", options=pos_view["Ticker"].tolist())
    with del_col2:
        if st.button("🗑️ Eliminar activo"):
            try:
                ok = delete_ticker_all_rows(del_ticker)
                if ok:
                    st.success(f"Se eliminaron las transacciones de {del_ticker} en Google Sheets.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("No se encontraron filas para eliminar (ya no existían).")
            except Exception as e:
                st.error(f"No se pudo eliminar: {e}")

elif menu == "Evaluar nueva acción (básico)":
    st.markdown("<h1>Evaluar nueva acción (básico)</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Compara tu portafolio actual vs. agregar un ticker con acciones propuestas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    tx = read_transactions_df()
    if tx.empty:
        st.warning("No hay transacciones en Google Sheets. Agrega compras en la sección 'Mi Portafolio'.")
        st.stop()

    c1,c2,c3,c4 = st.columns([1.2,1,1,1])
    with c1: new_ticker = st.text_input("Ticker candidato", "NFLX").upper().strip()
    with c2: shares_new = st.number_input("Acciones a comprar", min_value=0.0, value=5.0, step=1.0)
    with c3: bench = st.selectbox("Benchmark (β)", options=["SPY","^GSPC","QQQ"], index=0)
    with c4: rf = st.number_input("Tasa libre de riesgo", value=0.04, step=0.01)

    tickers = sorted(tx["Ticker"].unique().tolist())
    prices = load_prices(sorted(list(set(tickers + [new_ticker, bench]))), period="1y", interval="1d")
    if prices.empty:
        st.warning("No hay datos de precios para evaluar.")
        st.stop()

    last_prices = {t: get_last_close(t) for t in prices.columns}
    tx["LastPrice"] = tx["Ticker"].map(last_prices)
    tx["MktValue"] = tx["Shares"] * tx["LastPrice"]
    pos = tx.groupby("Ticker", as_index=False).agg({"Shares":"sum","MktValue":"sum"})

    shares_map = pos.set_index("Ticker")["Shares"].to_dict()
    aligned = prices.copy().reindex(columns=[c for c in prices.columns if c in shares_map], fill_value=np.nan)
    for t in aligned.columns:
        aligned[t] = aligned[t] * shares_map.get(t, 0.0)
    port_value = aligned.sum(axis=1).dropna()
    port_ret = port_value.pct_change().dropna()

    shares_map_after = shares_map.copy()
    shares_map_after[new_ticker] = shares_map_after.get(new_ticker, 0.0) + shares_new
    aligned_after = prices.copy().reindex(columns=[c for c in prices.columns if c in shares_map_after], fill_value=np.nan)
    for t in aligned_after.columns:
        aligned_after[t] = aligned_after[t] * shares_map_after.get(t, 0.0)
    port_value_after = aligned_after.sum(axis=1).dropna()
    port_ret_after = port_value_after.pct_change().dropna()

    def ann_stats(r):
        if r.empty: return np.nan, np.nan, np.nan, pd.Series(dtype=float), np.nan
        mu = (1 + r.mean())**252 - 1
        sigma = r.std() * np.sqrt(252)
        sharpe = (mu - rf) / sigma if sigma > 0 else np.nan
        cum = (1 + r).cumprod()
        mdd = max_drawdown(cum)
        return mu, sigma, sharpe, cum, mdd

    mu_b, sig_b, sh_b, cum_b, mdd_b = ann_stats(port_ret)
    mu_a, sig_a, sh_a, cum_a, mdd_a = ann_stats(port_ret_after)

    delta_sharpe = (sh_a - sh_b) if (not np.isnan(sh_a) and not np.isnan(sh_b)) else np.nan
    delta_mdd = (mdd_a - mdd_b) if (not np.isnan(mdd_a) and not np.isnan(mdd_b)) else np.nan
    corr_candidate = np.nan
    if new_ticker in prices.columns:
        cand_ret = prices[new_ticker].pct_change().dropna()
        corr_candidate = np.corrcoef(
            cand_ret.reindex(port_ret.index).dropna(),
            port_ret.reindex(cand_ret.index).dropna()
        )[0,1] if not port_ret.empty else np.nan

    decision = "Considerar"; badge_class = "badge-yellow"
    cond_verde = (delta_sharpe >= 0.03) and (delta_mdd >= -0.02) and (np.isnan(corr_candidate) or corr_candidate <= 0.75)
    cond_rojo  = (np.isnan(delta_sharpe) or delta_sharpe <= 0.0)

    if cond_verde:
        decision = "Agregar"; badge_class = "badge-green"
    elif cond_rojo:
        decision = "No agregar"; badge_class = "badge-red"

    st.markdown(f'**Recomendación:** <span class="badge {badge_class}">{decision}</span>', unsafe_allow_html=True)
    with st.expander("Ver resumen"):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Δ Sharpe", f'{(delta_sharpe if not np.isnan(delta_sharpe) else 0):+.2f}')
        c2.metric("Δ Volatilidad", f'{(sig_a - sig_b)*100:+.2f}%')
        c3.metric("Δ Max Drawdown", f'{(delta_mdd if not np.isnan(delta_mdd) else 0)*100:+.2f}%')
        c4.metric("Correlación cand.", f'{corr_candidate:.2f}' if not np.isnan(corr_candidate) else "N/A")

    fig_comp = go.Figure()
    if not cum_b.empty:
        fig_comp.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values, mode="lines", name="Portafolio (antes)"))
    if not cum_a.empty:
        fig_comp.add_trace(go.Scatter(x=cum_a.index, y=cum_a.values, mode="lines", name="Portafolio (después)"))
    if new_ticker in prices.columns:
        cum_new = (1 + prices[new_ticker].pct_change().dropna()).cumprod()
        fig_comp.add_trace(go.Scatter(x=cum_new.index, y=cum_new.values, mode="lines", name=new_ticker, line=dict(dash="dot")))
    fig_comp.update_layout(title="Crecimiento de $1: antes vs. después", template="simple_white",
                           xaxis_title="", yaxis_title="Crecimiento de $1", height=420)
    st.plotly_chart(fig_comp, use_container_width=True)

elif menu == "Riesgo de Inversión":
    st.markdown("<h1>Riesgo de Inversión</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Beta, alpha y varianza vs. un índice de referencia.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,2,1])
    with c1: stonk = st.text_input("Símbolo", "NVDA").upper()
    with c2: market = st.text_input("Índice de referencia (p. ej., ^GSPC)", "^GSPC").upper()
    with c3: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))

    period, interval = range_to_yf_params(range_key)
    df_stock = get_history(stonk, period, interval)
    df_market = get_history(market, period, interval)

    if df_stock.empty or df_market.empty:
        st.warning("No se pudieron obtener los datos.")
    else:
        s_ret, _, _ = return_metrics(df_stock)
        m_ret, _, _ = return_metrics(df_market)
        beta, alpha = compute_beta_alpha(s_ret["Return"], m_ret["Return"])
        var_market = m_ret["Return"].var()

        c1,c2,c3 = st.columns(3)
        c1.metric("Beta", f"{beta:.3f}")
        c2.metric("Varianza mercado", f"{var_market:.6f}")
        c3.metric("Alpha", f"{alpha:.4f}")

        st.subheader("Relación de rendimientos (acción vs mercado)")
        st.scatter_chart(pd.DataFrame({"Stock": s_ret["Return"], "Market": m_ret["Return"]}).dropna())

elif menu == "CAPM":
    st.markdown("<h1>CAPM</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Rendimiento esperado en función de β y del premio por riesgo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: rf = st.number_input("Tasa libre de riesgo (0.04 = 4%)", value=0.04)
    with c2: rm = st.number_input("Rendimiento esperado del mercado (0.10 = 10%)", value=0.10)
    with c3: use_data_beta = st.toggle("Estimar β desde datos", value=False)

    if use_data_beta:
        c1,c2,c3 = st.columns([2,2,1])
        with c1: stonk = st.text_input("Ticker del activo", "AAPL").upper()
        with c2: market = st.text_input("Benchmark (p. ej. ^GSPC)", "^GSPC").upper()
        with c3: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))
        period, interval = range_to_yf_params(range_key)
        s = get_history(stonk, period, interval)
        m = get_history(market, period, interval)
        if s.empty or m.empty:
            st.warning("No se pudo estimar β (datos insuficientes). Se usará β=1.0.")
            beta = 1.0
        else:
            s_ret, _, _ = return_metrics(s)
            m_ret, _, _ = return_metrics(m)
            beta, _ = compute_beta_alpha(s_ret["Return"], m_ret["Return"])
            if np.isnan(beta): beta = 1.0
    else:
        beta = st.number_input("β del activo", value=1.2)

    expected_return = rf + beta * (rm - rf)
    st.metric("Rendimiento esperado (CAPM)", f"{expected_return*100:.2f}%")

    betas = np.linspace(0,2,50)
    returns = rf + betas * (rm - rf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=returns, mode="lines", name="Línea SML"))
    fig.add_trace(go.Scatter(x=[beta], y=[expected_return], mode="markers+text",
                             text=["Tu activo"], textposition="bottom center",
                             marker=dict(size=10,color="red"), name="Activo"))
    fig.update_layout(title="Security Market Line (SML)", xaxis_title="Beta", yaxis_title="Rendimiento esperado",
                      template="simple_white", height=520, margin=dict(l=40,r=20,t=40,b=30))
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Optimización de Portafolio (Markowitz)":
    st.markdown("<h1>Optimización de Portafolio (Markowitz)</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Análisis libre por tickers y, debajo, análisis con tu portafolio desde Sheets.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ========= Sección 1: ANÁLISIS LIBRE (elige tus tickers) =========
    st.subheader("Análisis libre (elige tus tickers)")
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        free_tickers = [t.strip().upper() for t in st.text_input(
            "Tickers (separados por coma)",
            "AAPL, MSFT, NVDA, META, AMZN"
        ).split(",") if t.strip()]
    with c2:
        period_key_free = st.selectbox("Ventana de datos", ["1 año","3 años","5 años"], index=1, key="free_period")
    with c3:
        obj_free = st.selectbox("Objetivo", ["Máximo Sharpe","Mínima Varianza"], index=0, key="free_obj")
    with c4:
        rf_free = st.number_input("Rf", value=0.04, step=0.01, key="free_rf")

    c5, c6, c7 = st.columns([1,1,1])
    with c5:
        n_sims_free = st.slider("N.º simulaciones", 1000, 15000, 4000, step=500, key="free_nsims")
    with c6:
        shrink_alpha_free = st.slider("Shrinkage cov (α)", 0.0, 0.5, 0.10, step=0.02, key="free_alpha")
    with c7:
        per_min = st.number_input("Mín. por activo (%)", 0.0, 50.0, 2.0, step=0.5, key="free_min")/100
        per_max = st.number_input("Máx. por activo (%)", 1.0, 100.0, 30.0, step=1.0, key="free_max")/100

    if st.button("Ejecutar análisis libre"):
        period_map = {"1 año":"1y","3 años":"3y","5 años":"5y"}
        prices = prices_for_universe(free_tickers, period=period_map[period_key_free], interval="1d")
        rets = returns_matrix(prices)
        if rets.empty or rets.shape[1] < 2:
            st.warning("No hay datos suficientes para esos tickers.")
        else:
            mean_ret = rets.mean() * 252
            cov = shrink_cov(rets.cov() * 252, shrink_alpha_free)
            classes = {a: "Otros" for a in free_tickers}
            res, w_list = simulate_portfolios(mean_ret, cov, n_sims_free, per_min, per_max, group_limits=None, classes=classes)
            if res.empty:
                st.error("No se generaron portafolios que cumplan las restricciones. Ajusta límites.")
            else:
                idx = np.nanargmax(res["sharpe"].values) if obj_free=="Máximo Sharpe" else np.nanargmin(res["vol"].values)
                w_opt = pd.Series(w_list[idx], index=mean_ret.index); w_opt = w_opt/w_opt.sum()
                mu_opt = float(np.dot(w_opt.values, mean_ret.values))
                vol_opt = float(np.sqrt(w_opt.values @ cov.values @ w_opt.values))
                sharpe_opt = (mu_opt - rf_free) / vol_opt if vol_opt>0 else np.nan

                st.metric("Retorno esperado", f"{mu_opt*100:.2f}%")
                st.metric("Riesgo anualizado", f"{vol_opt*100:.2f}%")
                st.metric("Sharpe", f"{sharpe_opt:.2f}")

                st.subheader("Pesos propuestos")
                st.dataframe((w_opt.sort_values(ascending=False)*100).round(2).to_frame("Peso (%)"),
                             use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res["vol"], y=res["ret"], mode="markers",
                    marker=dict(color=res["sharpe"], colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
                    name="Portafolios simulados",
                    hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(x=[vol_opt], y=[mu_opt], mode="markers+text",
                                         text=["Óptimo"], textposition="bottom center",
                                         marker=dict(color="red", size=10), name="Óptimo"))
                fig.update_layout(title=f"Frontera simulada (Análisis libre) · {period_key_free}",
                                  xaxis_title="Riesgo (σ anualizado)", yaxis_title="Retorno esperado anualizado",
                                  template="simple_white", height=520)
                st.plotly_chart(fig, use_container_width=True)

    # ========= Sección 2: EXPANDER con MI PORTAFOLIO (Equity-only y ETFs-only) =========
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    with st.expander("Realizar análisis con mi portafolio"):
        tx = read_transactions_df()
        if tx.empty:
            st.warning("No hay transacciones en Google Sheets. Ve a “Mi Portafolio” y agrega compras.")
        else:
            tab_eq, tab_etf = st.tabs(["Equity-only", "ETFs-only"])

            # -------- Equity-only (excluye ETFs amplios) --------
            with tab_eq:
                st.caption("Optimiza solo acciones presentes en tu hoja (excluye SPY/VOO/IVV/QQQ/IWM).")

                cA1, cA2, cA3, cA4 = st.columns([1,1,1,1])
                with cA1:
                    period_key = st.selectbox("Ventana de datos", ["1 año","3 años","5 años"], index=1, key="eq_period")
                with cA2:
                    obj = st.selectbox("Objetivo", ["Máximo Sharpe","Mínima Varianza"], index=0, key="eq_obj")
                with cA3:
                    n_sims = st.slider("N.º simulaciones", 1000, 15000, 4000, step=500, key="eq_nsims")
                with cA4:
                    shrink_alpha = st.slider("Shrinkage cov (α)", 0.0, 0.5, 0.10, step=0.02, key="eq_alpha")

                cA5, cA6, cA7, cA8 = st.columns([1,1,1,1])
                with cA5:
                    amin = st.number_input("Mín. por activo (%)", 0.0, 50.0, 2.0, step=0.5, key="eq_min")/100
                with cA6:
                    amax = st.number_input("Máx. por activo (%)", 1.0, 100.0, 15.0, step=1.0, key="eq_max")/100
                with cA7:
                    tech_min = st.number_input("Tech mínimo (%)", 0.0, 100.0, 30.0, step=1.0, key="eq_tmin")/100
                with cA8:
                    tech_max = st.number_input("Tech máximo (%)", 0.0, 100.0, 50.0, step=1.0, key="eq_tmax")/100
                cA9, cA10 = st.columns([1,1])
                with cA9:
                    defens_min = st.number_input("Defensivo mínimo (%)", 0.0, 100.0, 20.0, step=1.0, key="eq_dmin")/100
                with cA10:
                    defens_max = st.number_input("Defensivo máximo (%)", 0.0, 100.0, 40.0, step=1.0, key="eq_dmax")/100

                if st.button("Optimizar mi Portafolio (Equity-only)"):
                    all_tickers = sorted(tx["Ticker"].unique().tolist())
                    equities = [t for t in all_tickers if t.upper() not in INDEX_ETF]
                    if len(equities) < 2:
                        st.warning("Necesito al menos 2 acciones en tu portafolio para optimizar.")
                    else:
                        w_cur = current_weights_from_tx(tx, equities)
                        if w_cur.empty or w_cur.sum() == 0:
                            st.warning("No pude construir pesos actuales del sleeve de acciones.")
                        else:
                            period_map = {"1 año":"1y","3 años":"3y","5 años":"5y"}
                            prices = prices_for_universe(equities, period=period_map[period_key], interval="1d")
                            rets = returns_matrix(prices)
                            if rets.empty or rets.shape[1] < 2:
                                st.warning("Datos insuficientes para calcular riesgo/retorno de las acciones.")
                            else:
                                mean_ret = rets.mean() * 252
                                cov = shrink_cov(rets.cov() * 252, shrink_alpha)
                                classes = {a: classify_ticker(a) for a in equities}
                                group_limits = {
                                    "Tecnología (alto beta)": (tech_min, tech_max),
                                    "Defensivo (bajo beta)": (defens_min, defens_max),
                                }
                                res, w_list = simulate_portfolios(mean_ret.loc[equities], cov.loc[equities, equities],
                                                                  n_sims=n_sims,
                                                                  per_asset_min=amin, per_asset_max=amax,
                                                                  group_limits=group_limits, classes=classes)
                                if res.empty:
                                    st.error("No se generaron portafolios que cumplan las restricciones. Ajusta límites.")
                                else:
                                    idx = np.nanargmax(res["sharpe"].values) if obj=="Máximo Sharpe" else np.nanargmin(res["vol"].values)
                                    w_opt = pd.Series(w_list[idx], index=equities); w_opt = w_opt/w_opt.sum()

                                    rf_local = st.number_input("Rf para métricas (Equity-only)", value=0.04, step=0.01, key="rf_eq_tab")
                                    mu_opt = float(np.dot(w_opt.values, mean_ret.loc[equities].values))
                                    vol_opt = float(np.sqrt(w_opt.values @ cov.loc[equities, equities].values @ w_opt.values))
                                    sharpe_opt = (mu_opt - rf_local) / vol_opt if vol_opt>0 else np.nan

                                    mu_cur = float(np.dot(w_cur.values, mean_ret.loc[equities].reindex(w_cur.index).values))
                                    vol_cur = float(np.sqrt(w_cur.values @ cov.loc[w_cur.index, w_cur.index].values @ w_cur.values))
                                    sharpe_cur = (mu_cur - rf_local) / vol_cur if vol_cur>0 else np.nan

                                    c1,c2,c3 = st.columns(3)
                                    c1.metric("Retorno (act → ópt)", f"{mu_cur*100:.2f}% → {mu_opt*100:.2f}%")
                                    c2.metric("Riesgo (act → ópt)", f"{vol_cur*100:.2f}% → {vol_opt*100:.2f}%")
                                    c3.metric("Sharpe (act → ópt)", f"{sharpe_cur:.2f} → {sharpe_opt:.2f}")

                                    comp = compare_weights_table(w_cur, w_opt)
                                    st.subheader("Pesos en acciones (normalizados a 100%)")
                                    st.dataframe(comp.style.format({"Actual":"{:.2f}%", "Propuesto":"{:.2f}%", "Δ (pp)":"{:.2f}"}),
                                                 use_container_width=True)
                                    st.caption(f"Turnover estimado (acciones): {turnover_abs(w_cur, w_opt)*100:.2f}%")

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=res["vol"], y=res["ret"], mode="markers",
                                        marker=dict(color=res["sharpe"], colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
                                        name="Portafolios simulados",
                                        hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
                                    ))
                                    fig.add_trace(go.Scatter(x=[vol_cur], y=[mu_cur], mode="markers+text",
                                                             text=["Actual"], textposition="top center",
                                                             marker=dict(color="orange", size=10), name="Actual"))
                                    fig.add_trace(go.Scatter(x=[vol_opt], y=[mu_opt], mode="markers+text",
                                                             text=["Óptimo"], textposition="bottom center",
                                                             marker=dict(color="red", size=10), name="Óptimo"))
                                    fig.update_layout(title=f"Frontera simulada (Equity-only) · {period_key}",
                                                      xaxis_title="Riesgo (σ anualizado)", yaxis_title="Retorno esperado anualizado",
                                                      template="simple_white", height=520)
                                    st.plotly_chart(fig, use_container_width=True)

            # -------- ETFs-only --------
            with tab_etf:
                st.caption("Optimiza solo ETFs amplios presentes en tu hoja (SPY/VOO/IVV/QQQ/IWM).")

                cB1, cB2, cB3, cB4 = st.columns([1,1,1,1])
                with cB1:
                    period_key2 = st.selectbox("Ventana de datos", ["1 año","3 años","5 años"], index=1, key="etf_period")
                with cB2:
                    obj2 = st.selectbox("Objetivo", ["Máximo Sharpe","Mínima Varianza"], index=0, key="etf_obj")
                with cB3:
                    n_sims2 = st.slider("N.º simulaciones", 1000, 15000, 4000, step=500, key="etf_nsims")
                with cB4:
                    shrink_alpha2 = st.slider("Shrinkage cov (α)", 0.0, 0.5, 0.10, step=0.02, key="etf_alpha")

                cB5, cB6 = st.columns([1,1])
                with cB5:
                    etf_min = st.number_input("Mín. por ETF (%)", 0.0, 100.0, 0.0, step=0.5, key="etf_min")/100
                with cB6:
                    etf_max = st.number_input("Máx. por ETF (%)", 1.0, 100.0, 100.0, step=1.0, key="etf_max")/100

                if st.button("Optimizar Asset Mix (ETFs)"):
                    all_tickers = sorted(tx["Ticker"].unique().tolist())
                    etfs = [t for t in all_tickers if t.upper() in INDEX_ETF]
                    if len(etfs) < 2:
                        st.warning("Se requieren al menos 2 ETFs para este análisis.")
                    else:
                        w_cur = current_weights_from_tx(tx, etfs)
                        if w_cur.empty or w_cur.sum() == 0:
                            st.warning("No pude construir pesos actuales de ETFs.")
                        else:
                            period_map = {"1 año":"1y","3 años":"3y","5 años":"5y"}
                            prices = prices_for_universe(etfs, period=period_map[period_key2], interval="1d")
                            rets = returns_matrix(prices)
                            if rets.empty or rets.shape[1] < 2:
                                st.warning("Datos insuficientes para calcular riesgo/retorno de los ETFs.")
                            else:
                                mean_ret = rets.mean() * 252
                                cov = shrink_cov(rets.cov() * 252, shrink_alpha2)
                                classes = {a: "Índice/ETF" for a in etfs}
                                res, w_list = simulate_portfolios(mean_ret.loc[etfs], cov.loc[etfs, etfs],
                                                                  n_sims=n_sims2,
                                                                  per_asset_min=etf_min, per_asset_max=etf_max,
                                                                  group_limits=None, classes=classes)
                                if res.empty:
                                    st.error("No se generaron portafolios que cumplan las restricciones. Ajusta límites.")
                                else:
                                    idx = np.nanargmax(res["sharpe"].values) if obj2=="Máximo Sharpe" else np.nanargmin(res["vol"].values)
                                    w_opt = pd.Series(w_list[idx], index=etfs); w_opt = w_opt/w_opt.sum()

                                    rf_local = st.number_input("Rf para métricas (ETFs)", value=0.04, step=0.01, key="rf_etf_tab")
                                    mu_opt = float(np.dot(w_opt.values, mean_ret.loc[etfs].values))
                                    vol_opt = float(np.sqrt(w_opt.values @ cov.loc[etfs, etfs].values @ w_opt.values))
                                    sharpe_opt = (mu_opt - rf_local) / vol_opt if vol_opt>0 else np.nan

                                    mu_cur = float(np.dot(w_cur.values, mean_ret.loc[etfs].reindex(w_cur.index).values))
                                    vol_cur = float(np.sqrt(w_cur.values @ cov.loc[w_cur.index, w_cur.index].values @ w_cur.values))
                                    sharpe_cur = (mu_cur - rf_local) / vol_cur if vol_cur>0 else np.nan

                                    c1,c2,c3 = st.columns(3)
                                    c1.metric("Retorno (act → ópt)", f"{mu_cur*100:.2f}% → {mu_opt*100:.2f}%")
                                    c2.metric("Riesgo (act → ópt)", f"{vol_cur*100:.2f}% → {vol_opt*100:.2f}%")
                                    c3.metric("Sharpe (act → ópt)", f"{sharpe_cur:.2f} → {sharpe_opt:.2f}")

                                    comp = compare_weights_table(w_cur, w_opt)
                                    st.subheader("Pesos en ETFs (normalizados a 100%)")
                                    st.dataframe(comp.style.format({"Actual":"{:.2f}%", "Propuesto":"{:.2f}%", "Δ (pp)":"{:.2f}"}),
                                                 use_container_width=True)
                                    st.caption(f"Turnover estimado (ETFs): {turnover_abs(w_cur, w_opt)*100:.2f}%")

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=res["vol"], y=res["ret"], mode="markers",
                                        marker=dict(color=res["sharpe"], colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
                                        name="Portafolios simulados",
                                        hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
                                    ))
                                    fig.add_trace(go.Scatter(x=[vol_cur], y=[mu_cur], mode="markers+text",
                                                             text=["Actual"], textposition="top center",
                                                             marker=dict(color="orange", size=10), name="Actual"))
                                    fig.add_trace(go.Scatter(x=[vol_opt], y=[mu_opt], mode="markers+text",
                                                             text=["Óptimo"], textposition="bottom center",
                                                             marker=dict(color="red", size=10), name="Óptimo"))
                                    fig.update_layout(title=f"Frontera simulada (ETFs) · {period_key2}",
                                                      xaxis_title="Riesgo (σ anualizado)", yaxis_title="Retorno esperado anualizado",
                                                      template="simple_white", height=520)
                                    st.plotly_chart(fig, use_container_width=True)

elif menu == "Simulación Monte Carlo":
    st.markdown("<h1>Simulación Monte Carlo</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Trayectorias de precio a 1 año (GBM) basadas en μ y σ estimados.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stonk = st.text_input("Símbolo", "AAPL").upper()
    with c2: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))
    with c3: simulations = st.slider("N.º de simulaciones", 50, 2000, 300, step=50)

    period, interval = range_to_yf_params(range_key)
    df = get_history(stonk, period, interval)
    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        df_ret, ann_ret, ann_vol = return_metrics(df)
        S0 = df_ret["Close"].iloc[-1]
        T, N = 1, 252
        np.random.seed(42)
        dt = T/N
        price_paths = np.zeros((N, simulations)); price_paths[0] = S0
        for t in range(1, N):
            rand = np.random.standard_normal(simulations)
            price_paths[t] = price_paths[t-1] * np.exp((ann_ret - 0.5*ann_vol**2)*dt + ann_vol*np.sqrt(dt)*rand)

        fig = go.Figure()
        for i in range(simulations):
            fig.add_trace(go.Scatter(y=price_paths[:,i], mode="lines", line=dict(width=0.7), showlegend=False))
        fig.update_layout(title=f"Simulación Monte Carlo de {stonk} · μ={ann_ret:.2%}, σ={ann_vol:.2%} · {range_key}",
                          xaxis_title="Días", yaxis_title="Precio simulado",
                          template="simple_white", height=680, margin=dict(l=40,r=20,t=48,b=30))
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto académico – Ingeniería Financiera")

