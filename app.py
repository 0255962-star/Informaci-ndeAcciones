import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# === Opcional Sheets (instaladas en requirements): gspread + google-auth
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except Exception:
    SHEETS_AVAILABLE = False

# =====================================================
# CONFIG BÁSICA
# =====================================================
st.set_page_config(page_title="Consulta de Acciones", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
.main .block-container {max-width: 1200px; padding-top: 0.5rem; padding-bottom: 2rem;}
h1,h2,h3 { font-weight: 700; letter-spacing: -0.2px; }
.section-subtitle { color:#6b7280; margin-bottom: 1rem; }
.hr { border:none; border-top:1px solid #e5e7eb; margin:1rem 0 1.25rem; }
.stSidebar { border-right:1px solid #e5e7eb; }
.sidebar-title { font-weight:700; font-size:1.05rem; margin:0.25rem 0 0.5rem; }
.small { color:#6b7280; font-size:0.92rem; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; height:92px; }
.stat-label { color:#6b7280; font-size:0.85rem; margin-bottom: 0.15rem; }
.stat-value { font-weight:700; font-size:1.1rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown('<div class="sidebar-title">Menú</div>', unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Consulta de Acciones",
        "Mi Portafolio (Google Sheets)",
        "Riesgo de Inversión",
        "CAPM",
        "Optimización de Portafolio (Markowitz)",
        "Simulación Monte Carlo",
    ],
)

# =====================================================
# SELECTOR DE PERIODO
# =====================================================
RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "1 año", "YTD", "3 años", "5 años"]

def range_to_yf_params(key: str):
    return {
        "1 semana": ("7d", "1d"),
        "1 mes": ("1mo", "1d"),
        "6 meses": ("6mo", "1d"),
        "1 año": ("1y", "1d"),
        "YTD": ("ytd", "1d"),
        "3 años": ("3y", "1wk"),
        "5 años": ("5y", "1wk"),
    }.get(key, ("6mo", "1d"))

# =====================================================
# DATOS (Yahoo Finance)
# =====================================================
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
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df.dropna(subset=["Open","High","Low","Close"])

def add_sma(df: pd.DataFrame, windows=(20,50,200)):
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out

@st.cache_data(ttl=1800)
def load_prices(tickers, period="1y", interval="1d") -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()

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
            prices = raw[["Adj Close"]].copy()
            prices.columns = [tickers[0]]
        else:
            return pd.DataFrame()
    return prices.dropna(how="all")

def returns_annualized(df_prices: pd.DataFrame):
    rets = df_prices.pct_change().dropna()
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    return rets, ann_ret, ann_vol

def compute_beta_alpha(stock_returns: pd.Series, market_returns: pd.Series):
    merged = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if merged.shape[0] < 2:
        return np.nan, np.nan
    cov = np.cov(merged.iloc[:,0], merged.iloc[:,1])[0][1]
    var_m = merged.iloc[:,1].var()
    beta = cov / var_m if var_m != 0 else np.nan
    alpha = merged.iloc[:,0].mean() - beta * merged.iloc[:,1].mean()
    return beta, alpha

def human_number(n):
    try:
        n = float(n)
    except Exception:
        return "N/A"
    if abs(n) >= 1e12: return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

def pct(x):
    try: return f"{float(x)*100:.2f}%"
    except Exception: return "N/A"

def ratio(x):
    try: return f"{float(x):.2f}"
    except Exception: return "N/A"

# =====================================================
# PLOTLY – velas pro
# =====================================================
TEMPLATE = "simple_white"
COLOR_UP, COLOR_UP_FILL = "rgba(22,163,74,1)", "rgba(22,163,74,0.9)"
COLOR_DOWN, COLOR_DOWN_FILL = "rgba(220,38,38,1)", "rgba(220,38,38,0.9)"
SMA_COLORS = {"SMA20":"#5546d6","SMA50":"#f59e0b","SMA200":"#b78d0a"}

def plot_candles_with_volume(df: pd.DataFrame, title: str,
                             show_sma20=True, show_sma50=True, show_sma200=True):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.72, 0.28])

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color=COLOR_UP,
            increasing_fillcolor=COLOR_UP_FILL,
            decreasing_line_color=COLOR_DOWN,
            decreasing_fillcolor=COLOR_DOWN_FILL,
            line=dict(width=1.2), whiskerwidth=0.5, opacity=0.95,
        ),
        row=1, col=1
    )

    for k, flag in [("SMA20",show_sma20),("SMA50",show_sma50),("SMA200",show_sma200)]:
        if flag and k in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[k], mode="lines",
                           line=dict(color=SMA_COLORS[k], width=1.6), name=k),
                row=1, col=1
            )

    vol_colors = [COLOR_UP_FILL if c>=o else COLOR_DOWN_FILL for o,c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volumen", opacity=0.85),
                  row=2, col=1)

    fig.update_xaxes(
        showgrid=False,
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(step="year", stepmode="todate", label="YTD"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        rangebreaks=[dict(bounds=["sat","mon"])],
        row=1, col=1
    )
    fig.update_xaxes(showgrid=False, rangebreaks=[dict(bounds=["sat","mon"])], row=2, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Volumen", showgrid=True, zeroline=False, row=2, col=1)

    fig.update_layout(
        title=title, template=TEMPLATE, height=760,
        margin=dict(l=40, r=20, t=48, b=30),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
    )
    return fig

# =====================================================
# GEMINI – Traducción
# =====================================================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
BASES = ["https://generativelanguage.googleapis.com/v1",
         "https://generativelanguage.googleapis.com/v1beta"]
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
    last = None
    for base in BASES:
        ok, data = list_models(base)
        if not ok:
            last = data; continue
        models = data.get("models", []) if isinstance(data, dict) else []
        supported = set()
        for m in models:
            name = m.get("name"); methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
            if not name: continue
            short = name.split("/")[-1]
            if any(x in ("generateContent","generate_content") for x in methods):
                supported.add(short)
        for pref in PREFERRED:
            if pref in supported: return base, pref
        if supported: return base, sorted(supported)[0]
        last = "No models with generateContent"
    return None, None

def generate_content_rest(base: str, model: str, text: str) -> str:
    url = f"{base}/models/{model}:generateContent"
    payload = {"contents":[{"parts":[{"text":text}]}]}
    r = requests.post(url, params={"key": API_KEY},
                      headers={"Content-Type":"application/json"},
                      data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        try: err = r.json()
        except Exception: err = r.text
        raise RuntimeError(f"{r.status_code} {err}")
    data = r.json()
    for cand in data.get("candidates", []):
        content = cand.get("content") or {}; parts = content.get("parts") or []
        for p in parts:
            t = (p.get("text") or "").strip()
            if t: return t
    return ""

def translate_to_spanish(text: str) -> str:
    if not text or not API_KEY: return ""
    base, model = pick_model_and_base()
    if not base or not model: raise RuntimeError("Gemini no disponible. Revisa GOOGLE_API_KEY.")
    prompt = ("Traduce al español el siguiente texto sobre una empresa. "
              "Usa español claro y natural de negocios. Conserva nombres propios y números. "
              "Solo devuelve la traducción, sin explicaciones.\n\n" + text)
    return generate_content_rest(base, model, prompt)

# =====================================================
# GOOGLE SHEETS
# =====================================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

def parse_spreadsheet_target(text: str):
    if not text: return None, None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if m: return None, m.group(1)
    if re.fullmatch(r"[A-Za-z0-9\-_]{20,}", text):  # ID puro
        return None, text
    return text, None  # nombre

@st.cache_resource
def get_gs_client():
    if not SHEETS_AVAILABLE: return None
    info = st.secrets.get("gcp_service_account", None)
    if info:
        creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
        return gspread.authorize(creds)
    if os.path.exists("credentials.json"):
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        return gspread.authorize(creds)
    return None

@st.cache_data(ttl=300)
def read_portfolio_from_sheet(spreadsheet_target: str, worksheet_name: str) -> pd.DataFrame:
    client = get_gs_client()
    if client is None: return pd.DataFrame()
    title, key = parse_spreadsheet_target(spreadsheet_target)
    try:
        sh = client.open_by_key(key) if key else client.open(title)
        ws = sh.worksheet(worksheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# =====================================================
# CLASIFICACIÓN SIMPLE
# =====================================================
TECH_HIGH = {"NVDA","GOOG","GOOGL","META","MSFT","AMZN","TSLA","AMD","AVGO"}
LOW_VOL = {"KO","PG","JNJ","PEP","COST","WMT","MCD","HD","MRK"}
INDEX_ETF = {"SPY","VOO","IVV","QQQ","IWM"}

def classify_ticker(t: str) -> str:
    t = t.upper()
    if t in INDEX_ETF: return "Índice/ETF"
    if t in TECH_HIGH: return "Tecnología (alto beta)"
    if t in LOW_VOL: return "Defensivo (bajo beta)"
    return "Otros"

# =====================================================
# 1) CONSULTA DE ACCIONES
# =====================================================
if menu == "Consulta de Acciones":
    st.markdown("<h1>Consulta de Acciones</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Descripción, métricas clave y velas interactivas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c0, c1, c2, c3, c4 = st.columns([1.5,1,1,1,1.2])
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
    info = getattr(ticker, "info", {}) or {}

    # Descripción al inicio
    st.subheader("Empresa")
    st.write(info.get("longName", "No disponible"))

    st.subheader("Descripción (inglés)")
    summary = info.get("longBusinessSummary", "")
    st.write(summary if summary else "No disponible.")

    col_tr = st.columns([1,3])[0]
    with col_tr:
        if st.button("Traducir a español"):
            if not API_KEY:
                st.warning("Configura GOOGLE_API_KEY en Secrets para usar Gemini.")
            elif not summary:
                st.warning("No hay descripción para traducir.")
            else:
                with st.spinner("Traduciendo con Gemini..."):
                    try:
                        st.session_state[f"translated_{stonk}"] = translate_to_spanish(summary)
                    except Exception as e:
                        st.error(f"Error al traducir con Gemini: {e}")

    if st.session_state.get(f"translated_{stonk}"):
        st.subheader("Descripción (español)")
        st.write(st.session_state[f"translated_{stonk}"])

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Métricas clave (10)
    st.subheader("Indicadores clave")
    key_data = {
        "Sector": info.get("sector","N/A"),
        "Industria": info.get("industry","N/A"),
        "País": info.get("country","N/A"),
        "Market Cap": human_number(info.get("marketCap")),
        "Beta (5y)": ratio(info.get("beta")),
        "P/E (TTM)": ratio(info.get("trailingPE")),
        "P/E (Fwd)": ratio(info.get("forwardPE")),
        "P/B": ratio(info.get("priceToBook")),
        "Div. Yield": pct(info.get("dividendYield")),
        "Cambio 52w": pct(info.get("52WeekChange")),
    }
    cols_top = st.columns(5); cols_bottom = st.columns(5)
    keys = list(key_data.keys())
    for i, col in enumerate(cols_top):
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{keys[i]}</div>'
                        f'<div class="stat-value">{key_data[keys[i]]}</div></div>', unsafe_allow_html=True)
    for i, col in enumerate(cols_bottom, start=5):
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{keys[i]}</div>'
                        f'<div class="stat-value">{key_data[keys[i]]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Velas
    st.subheader("Gráfica de velas con volumen")
    period, interval = range_to_yf_params(range_key)
    hist = get_history(stonk, period, interval)
    if hist.empty:
        st.warning("No se pudo obtener información histórica suficiente.")
    else:
        hist = add_sma(hist, (20,50,200))
        fig = plot_candles_with_volume(hist, f"{stonk} · {range_key}",
                                       show_sma20=show_sma20, show_sma50=show_sma50, show_sma200=show_sma200)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2) MI PORTAFOLIO (Google Sheets)
# =====================================================
elif menu == "Mi Portafolio (Google Sheets)":
    st.markdown("<h1>Mi Portafolio</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Conexión a Google Sheets, análisis de rendimiento, riesgo y contribuciones.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    colA, colB, colC = st.columns([2.2, 1.2, 1])
    with colA:
        spreadsheet_target = st.text_input(
            "Spreadsheet (nombre, URL o ID)",
            value=st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
        )
    with colB:
        worksheet = st.text_input(
            "Worksheet",
            value=st.secrets.get("GSHEETS_WORKSHEET_NAME", "Portafolio")
        )
    with colC:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))

    # Si no hay Sheets libs o credenciales:
    if not SHEETS_AVAILABLE:
        st.error("Faltan librerías para Google Sheets. Instala `gspread` y `google-auth`.")
        st.stop()

    # Leer sheet
    df_sheet = read_portfolio_from_sheet(spreadsheet_target, worksheet) if spreadsheet_target else pd.DataFrame()

    # Portafolio de ejemplo si falla lectura
    SAMPLE = pd.DataFrame({
        "Ticker": ["SPY","KO","PG","JNJ","PEP","COST","NVDA","GOOG","META","MSFT","AMZN"],
        "Weight": [0.30,0.08,0.07,0.06,0.05,0.04,0.12,0.09,0.08,0.06,0.05]
    })
    if df_sheet.empty or not {"Ticker","Weight"}.issubset(df_sheet.columns):
        st.warning("No se pudo leer el portafolio desde Sheets. Mostrando ejemplo (edita tu Sheets para reemplazarlo).")
        pf = SAMPLE.copy()
    else:
        pf = df_sheet[["Ticker","Weight"]].copy()
        pf["Ticker"] = pf["Ticker"].astype(str).str.upper()
        pf["Weight"] = pd.to_numeric(pf["Weight"], errors="coerce").fillna(0)

    # Normalizar pesos
    s = pf["Weight"].sum()
    if s <= 0:
        st.error("Los pesos de la hoja están vacíos o suman 0.")
        st.stop()
    pf["Weight"] = pf["Weight"] / s

    # Clasificación
    pf["Grupo"] = pf["Ticker"].apply(classify_ticker)

    # Datos de precios
    period, interval = range_to_yf_params(range_key)
    tickers = pf["Ticker"].tolist()
    prices = load_prices(tickers, period=period, interval=interval)
    if prices.empty:
        st.error("No se pudieron descargar precios para los tickers del portafolio.")
        st.stop()

    # Retornos y métricas
    rets = prices.pct_change().dropna()
    weights = pf.set_index("Ticker")["Weight"].reindex(rets.columns).fillna(0).values
    port_ret = rets.dot(weights)
    ann_ret = (1 + port_ret.mean())**252 - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    rf = st.number_input("Tasa libre de riesgo (para Sharpe)", value=0.04, step=0.01)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    # Beta vs benchmark
    bench = st.selectbox("Benchmark para β", options=["SPY","^GSPC","QQQ"], index=0)
    bench_prices = load_prices([bench], period=period, interval=interval)
    beta = np.nan
    if not bench_prices.empty:
        bench_ret = bench_prices.pct_change().dropna().iloc[:,0]
        merged = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
        if merged.shape[0] >= 2:
            cov = np.cov(merged.iloc[:,0], merged.iloc[:,1])[0][1]
            var_m = merged.iloc[:,1].var()
            beta = cov/var_m if var_m != 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendimiento anualizado", f"{ann_ret*100:.2f}%")
    c2.metric("Volatilidad anualizada", f"{ann_vol*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}")
    c4.metric("Beta vs " + bench, f"{beta:.2f}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Curva acumulada vs benchmark
    port_cum = (1+port_ret).cumprod()
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode="lines", name="Portafolio"))
    if not bench_prices.empty:
        br = bench_prices.pct_change().dropna().iloc[:,0]
        br = br.reindex(port_ret.index).fillna(method="pad")
        bench_cum = (1+br).cumprod()
        fig_perf.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode="lines", name=bench))
    fig_perf.update_layout(title=f"Rendimiento acumulado · {range_key}", template=TEMPLATE,
                           xaxis_title="", yaxis_title="Crecimiento de $1", height=420)
    st.plotly_chart(fig_perf, use_container_width=True)

    # Pie por activo
    pie = go.Figure(data=[go.Pie(labels=pf["Ticker"], values=pf["Weight"], hole=0.45)])
    pie.update_layout(title="Asignación por activo", template=TEMPLATE, height=420)
    st.plotly_chart(pie, use_container_width=True)

    # Contribución al riesgo
    cov_ann = rets.cov() * 252
    if cov_ann.shape[0] == len(weights):
        sigma_p = float(np.sqrt(weights.T @ cov_ann.values @ weights))
        if sigma_p > 0:
            mcontrib = cov_ann.values @ weights / sigma_p
            rcontrib = weights * mcontrib
            rc_df = pd.DataFrame({"Ticker": rets.columns, "RiskContribution": rcontrib})
            rc_df["Share"] = rc_df["RiskContribution"] / rc_df["RiskContribution"].sum()
            rc_df = rc_df.sort_values("Share", ascending=False)
            fig_rc = go.Figure(go.Bar(x=rc_df["Ticker"], y=rc_df["Share"]*100))
            fig_rc.update_layout(title="Contribución al riesgo (%)", template=TEMPLATE,
                                 xaxis_title="", yaxis_title="% de riesgo")
            st.plotly_chart(fig_rc, use_container_width=True)

    # Distribución por grupo
    grp = pf.groupby("Grupo", as_index=False)["Weight"].sum().sort_values("Weight", ascending=False)
    if not grp.empty:
        fig_grp = go.Figure(data=[go.Pie(labels=grp["Grupo"], values=grp["Weight"], hole=0.5)])
        fig_grp.update_layout(title="Distribución por grupo", template=TEMPLATE, height=400)
        st.plotly_chart(fig_grp, use_container_width=True)

    # Detalle
    indiv_ann_ret = (1 + rets.mean())**252 - 1
    indiv_ann_vol = rets.std() * np.sqrt(252)
    detail = pd.DataFrame({
        "Weight": pf.set_index("Ticker")["Weight"],
        "Grupo": pf.set_index("Ticker")["Grupo"],
        "AnnReturn": indiv_ann_ret,
        "AnnVol": indiv_ann_vol
    }).reindex(rets.columns)
    st.subheader("Detalle por activo")
    st.dataframe(detail.style.format({"Weight":"{:.2%}","AnnReturn":"{:.2%}","AnnVol":"{:.2%}"}),
                 use_container_width=True)

    st.info("Tip: en tu Google Sheet usa encabezados 'Ticker' y 'Weight'. Los pesos se normalizan automáticamente.")

# =====================================================
# 3) RIESGO DE INVERSIÓN
# =====================================================
elif menu == "Riesgo de Inversión":
    st.markdown("<h1>Riesgo de Inversión</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">β, α y relación de retornos vs índice.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,1])
    with c1: stonk = st.text_input("Símbolo", "NVDA").upper()
    with c2: market = st.text_input("Índice (p.ej., ^GSPC)", "^GSPC").upper()
    with c3: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))

    period, interval = range_to_yf_params(range_key)
    s = get_history(stonk, period, interval)
    m = get_history(market, period, interval)
    if s.empty or m.empty:
        st.warning("No se pudieron obtener datos suficientes.")
    else:
        sr = s["Close"].pct_change().dropna()
        mr = m["Close"].pct_change().dropna()
        beta, alpha = compute_beta_alpha(sr, mr)
        c1, c2, c3 = st.columns(3)
        c1.metric("Beta", f"{beta:.3f}")
        c2.metric("Var(Mercado)", f"{mr.var():.6f}")
        c3.metric("Alpha (media exc.)", f"{alpha:.4f}")
        st.subheader("Dispersión (Stock vs Mercado)")
        st.scatter_chart(pd.DataFrame({"Stock": sr, "Market": mr}).dropna())

# =====================================================
# 4) CAPM
# =====================================================
elif menu == "CAPM":
    st.markdown("<h1>CAPM</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Rendimiento esperado por β y premio por riesgo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: rf = st.number_input("Tasa libre de riesgo", value=0.04)
    with c2: rm = st.number_input("Rend. esperado de mercado", value=0.10)
    with c3: use_data = st.toggle("Estimar β desde datos", value=False)

    if use_data:
        d1,d2,d3 = st.columns([2,2,1])
        with d1: stonk = st.text_input("Ticker", "AAPL").upper()
        with d2: market = st.text_input("Benchmark", "^GSPC").upper()
        with d3: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))
        p,i = range_to_yf_params(range_key)
        s = get_history(stonk, p, i); m = get_history(market, p, i)
        if s.empty or m.empty:
            beta = 1.0; st.warning("Datos insuficientes; usando β=1.0")
        else:
            sr = s["Close"].pct_change().dropna(); mr = m["Close"].pct_change().dropna()
            beta,_ = compute_beta_alpha(sr, mr); beta = 1.0 if np.isnan(beta) else beta
    else:
        beta = st.number_input("β del activo", value=1.2)

    expected = rf + beta*(rm - rf)
    st.metric("Rendimiento esperado (CAPM)", f"{expected*100:.2f}%")

    xs = np.linspace(0,2,50); ys = rf + xs*(rm - rf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="SML"))
    fig.add_trace(go.Scatter(x=[beta], y=[expected], mode="markers+text",
                             text=["Tu activo"], textposition="bottom center",
                             marker=dict(size=10, color="red"), name="Activo"))
    fig.update_layout(title="Security Market Line", xaxis_title="Beta", yaxis_title="Rend. esperado",
                      template=TEMPLATE, height=520)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 5) MARKOWITZ
# =====================================================
elif menu == "Optimización de Portafolio (Markowitz)":
    st.markdown("<h1>Optimización de Portafolio (Markowitz)</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Frontera eficiente y portafolio de máximo Sharpe.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2 = st.columns([3,1])
    with c1:
        tickers_input = st.text_input("Tickers (coma)", "AAPL,MSFT,NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))

    p,i = range_to_yf_params(range_key)
    prices = load_prices(tickers, period=p, interval=i)
    if prices.empty or prices.shape[1] < 2:
        st.warning("Necesito al menos 2 tickers con datos.")
    else:
        rets = prices.pct_change().dropna()
        mean_returns = rets.mean()*252
        cov_matrix = rets.cov()*252

        c1,c2 = st.columns(2)
        with c1: nports = st.slider("Portafolios simulados", 1000, 10000, 3000, step=500)
        with c2: rf = st.number_input("Tasa libre de riesgo", value=0.04, step=0.01)

        results = np.zeros((3, nports))
        weights_rec = []
        rng = np.random.default_rng(42)
        for iport in range(nports):
            w = rng.random(len(tickers)); w /= w.sum()
            pret = np.dot(w, mean_returns)
            pvol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (pret - rf)/pvol if pvol>0 else np.nan
            results[:, iport] = [pvol, pret, sharpe]
            weights_rec.append(w)

        idx = np.nanargmax(results[2]); ms_vol, ms_ret, ms_sh = results[:, idx]; ms_w = weights_rec[idx]
        st.markdown(f"**Máx. Sharpe**: {ms_sh:.2f} · **Ret**: {ms_ret:.2%} · **Riesgo**: {ms_vol:.2%}")
        st.dataframe(pd.Series(ms_w, index=tickers, name="Pesos óptimos").to_frame().T, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results[0], y=results[1], mode="markers",
                                 marker=dict(color=results[2], colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
                                 name="Portafolios"))
        fig.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret], mode="markers+text", text=["Máx. Sharpe"],
                                 textposition="top center", marker=dict(color="red", size=10)))
        fig.update_layout(title=f"Frontera Eficiente · {range_key}",
                          xaxis_title="Riesgo (σ anual)", yaxis_title="Retorno anual esperado",
                          template=TEMPLATE, height=620)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6) MONTE CARLO
# =====================================================
elif menu == "Simulación Monte Carlo":
    st.markdown("<h1>Simulación Monte Carlo</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Trayectorias a 1 año usando μ y σ estimados.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stonk = st.text_input("Símbolo", "AAPL").upper()
    with c2: range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))
    with c3: sims = st.slider("Nº simulaciones", 50, 2000, 300, step=50)

    p,i = range_to_yf_params(range_key)
    df = get_history(stonk, p, i)
    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        ret = df["Close"].pct_change().dropna()
        ann_ret = ret.mean()*252
        ann_vol = ret.std()*np.sqrt(252)
        S0 = df["Close"].iloc[-1]; T=1; N=252; dt=T/N
        np.random.seed(42)
        paths = np.zeros((N, sims)); paths[0] = S0
        for t in range(1, N):
            z = np.random.standard_normal(sims)
            paths[t] = paths[t-1] * np.exp((ann_ret - 0.5*ann_vol**2)*dt + ann_vol*np.sqrt(dt)*z)

        fig = go.Figure()
        for k in range(sims):
            fig.add_trace(go.Scatter(y=paths[:,k], mode="lines", line=dict(width=0.7), showlegend=False))
        fig.update_layout(title=f"Monte Carlo {stonk} · μ={ann_ret:.2%}, σ={ann_vol:.2%} · {range_key}",
                          xaxis_title="Días", yaxis_title="Precio simulado",
                          template=TEMPLATE, height=680)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small">Proyecto académico – Ingeniería Financiera</div>', unsafe_allow_html=True)
