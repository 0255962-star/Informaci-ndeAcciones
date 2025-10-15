import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scraping
from bs4 import BeautifulSoup

st.set_page_config(page_title="Consulta de Acciones - GASCON", page_icon="📊")

# =========================
# 🔐 (OPCIONAL) Gemini para traducir textos
# =========================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

BASES = [
    "https://generativelanguage.googleapis.com/v1",       # preferido
    "https://generativelanguage.googleapis.com/v1beta",   # fallback
]
PREFERRED = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-pro",
]

def _list_models(base: str):
    url = f"{base}/models"
    try:
        r = requests.get(url, params={"key": API_KEY}, timeout=15)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json() if "application/json" in r.headers.get("content-type","") else r.text
    except Exception as e:
        return False, str(e)

def _pick_model_and_base():
    last_err = None
    for base in BASES:
        ok, data = _list_models(base)
        if not ok:
            last_err = data
            continue
        models = data.get("models", [])
        supported = set()
        for m in models:
            name = m.get("name")
            methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
            if not name:
                continue
            short = name.split("/")[-1]
            if any(x in ("generateContent", "generate_content") for x in methods):
                supported.add(short)
        for pref in PREFERRED:
            if pref in supported:
                return base, pref
        if supported:
            return base, sorted(supported)[0]
        last_err = "No hay modelos con generateContent en este endpoint."
    raise RuntimeError(f"No se pudo seleccionar modelo/base. Último error: {last_err}")

def _generate_content_rest(base: str, model: str, text: str) -> str:
    url = f"{base}/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": text}]}]}
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    r = requests.post(url, params=params, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"{r.status_code} {err}")
    data = r.json()
    for cand in data.get("candidates", []):
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            txt = (p.get("text") or "").strip()
            if txt:
                return txt
    return ""

def translate_to_spanish(text: str) -> str:
    if not text or not API_KEY:
        return ""
    try:
        base, model = _pick_model_and_base()
    except Exception:
        return ""
    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n" + text
    )
    return _generate_content_rest(base, model, prompt)

# =========================
# 🥄 Scraping con BeautifulSoup (Finviz)
# =========================
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=1800)
def scrape_finviz_snapshot(ticker: str) -> dict:
    """
    Extrae la tabla de 'snapshot' de Finviz (fundamentales y ratios clave).
    Devuelve un dict {campo: valor}.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", class_="snapshot-table2")
    if not table:
        return {}

    data = {}
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        for i in range(0, len(cells) - 1, 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            data[key] = value
    return data

@st.cache_data(ttl=1800)
def scrape_finviz_company_name(ticker: str) -> str:
    """
    Intenta extraer el nombre de la empresa del bloque de título en Finviz.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return ""
    soup = BeautifulSoup(r.text, "lxml")
    title_tbl = soup.find("table", class_="fullview-title")
    if not title_tbl:
        return ""
    # Heurística: suele estar en <b> o en un <a> dentro
    bold = title_tbl.find("b")
    if bold and bold.get_text(strip=True):
        return bold.get_text(strip=True)
    # fallback
    a = title_tbl.find("a")
    if a and a.get_text(strip=True):
        return a.get_text(strip=True)
    return ""

# =========================
# ⛽️ Datos históricos para velas (CSV estático de Stooq)
# =========================
def _stooq_symbol(symbol: str) -> list[str]:
    """
    Stooq suele usar tickers como 'aapl.us' para acciones USA.
    Probamos varias variantes para robustez.
    """
    base = symbol.lower()
    cands = [
        f"{base}.us",   # típico para USA
        base,           # por si ya viene con sufijo
        f"{base}.us",   # redundante — mantenido por claridad
    ]
    # Eliminar duplicados preservando orden
    seen, ordered = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

@st.cache_data(ttl=1800)
def fetch_stooq_history(symbol: str) -> pd.DataFrame:
    """
    Descarga histórico diario de Stooq como CSV y lo devuelve como DataFrame con
    columnas ['Open','High','Low','Close','Volume'] e índice DatetimeIndex.
    """
    candidates = _stooq_symbol(symbol)
    for s in candidates:
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text.splitlines()[0]:
                df = pd.read_csv(pd.compat.StringIO(r.text))
                # Compat: en algunas versiones, usar io.StringIO
                # df = pd.read_csv(io.StringIO(r.text))
                # Normalizar
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
                out = df[keep].dropna(subset=["Open","High","Low","Close"])
                if not out.empty:
                    return out
        except Exception:
            continue
    return pd.DataFrame()

# =========================
# 🎚️ Rango temporal + utilidades
# =========================
RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "YTD", "1 año", "3 años", "5 años"]

def _ytd_dates():
    today = date.today()
    start = date(today.year, 1, 1)
    return pd.Timestamp(start), pd.Timestamp(today)

def filter_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    if range_key == "1 semana":
        start = end - pd.Timedelta(days=7)
    elif range_key == "1 mes":
        start = end - pd.Timedelta(days=31)
    elif range_key == "6 meses":
        start = end - pd.Timedelta(days=186)
    elif range_key == "YTD":
        start, _ = _ytd_dates()
    elif range_key == "1 año":
        start = end - pd.Timedelta(days=365)
    elif range_key == "3 años":
        start = end - pd.Timedelta(days=365*3)
    elif range_key == "5 años":
        start = end - pd.Timedelta(days=365*5)
    else:
        start = df.index.min()
    return df.loc[df.index >= start]

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    res = pd.DataFrame()
    res["Open"]  = df["Open"].resample(rule).first()
    res["High"]  = df["High"].resample(rule).max()
    res["Low"]   = df["Low"].resample(rule).min()
    res["Close"] = df["Close"].resample(rule).last()
    if "Volume" in df.columns:
        res["Volume"] = df["Volume"].resample(rule).sum()
    return res.dropna(subset=["Open","High","Low","Close"])

def density_aware_downsample(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    n = len(df)
    if n > 800:
        return resample_ohlc(df, "M"), " (mensual)"
    if n > 300:
        return resample_ohlc(df, "W"), " (semanal)"
    return df, ""

def add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out

def volume_colors(df: pd.DataFrame):
    up = "rgba(22,163,74,0.75)"
    down = "rgba(220,38,38,0.75)"
    return [up if c >= o else down for o, c in zip(df["Open"], df["Close"])]

# =========================
# 🖥️ UI
# =========================
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR (BeautifulSoup + Finviz)")

# Entrada de símbolo
stonk = st.text_input("Ingresa el símbolo de la acción", "AAPL").strip().upper()

# Nombre empresa (Finviz)
company_name = scrape_finviz_company_name(stonk)
st.subheader("🏢 Nombre de la empresa")
st.write(company_name if company_name else "N/A")

# Datos fundamentales (Finviz)
fundamentals = scrape_finviz_snapshot(stonk)
st.subheader("📑 Datos fundamentales (Finviz)")
if fundamentals:
    df_f = pd.DataFrame([fundamentals]).T
    df_f.columns = [stonk]
    st.dataframe(df_f, use_container_width=True, height=400)
else:
    st.info("No se encontraron fundamentales (la estructura pudo cambiar o el ticker no existe en Finviz).")

# (Opcional) Bloque de traducción — usado si algún día agregas una descripción textual
st.write("---")
text_to_translate = ""
if text_to_translate and API_KEY:
    if st.button("Traducir a español 🇪🇸", use_container_width=True):
        with st.spinner("Traduciendo con Gemini..."):
            try:
                translated = translate_to_spanish(text_to_translate)
                if translated:
                    st.subheader("🇪🇸 Traducción")
                    st.write(translated)
            except Exception as e:
                st.error(f"Error al traducir con Gemini: {e}")

# =========================
# 📈 Gráfica de Velas (Plotly) con históricos Stooq
# =========================
st.write("---")
st.subheader("📈 Velas OHLC con Volumen (Stooq)")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    default_range = "6 meses"
    range_key = st.selectbox("Rango de tiempo", RANGE_OPTIONS, index=RANGE_OPTIONS.index(default_range))
with c2:
    auto_downsample = st.toggle("Agrupar automáticamente (W/M)", value=True)
with c3:
    show_sma = st.toggle("Mostrar SMA 20/50", value=True)

hist_all = fetch_stooq_history(stonk)

if hist_all.empty:
    st.warning("No se pudo obtener el histórico desde Stooq (intenta con otro ticker o revisa el sufijo).")
else:
    hist = filter_by_range(hist_all, range_key)

    if hist.empty:
        st.warning("No hay datos en el rango seleccionado.")
    else:
        suffix = ""
        df_plot = hist.copy()
        if auto_downsample:
            df_plot, suffix = density_aware_downsample(df_plot)
        if show_sma:
            df_plot = add_moving_averages(df_plot, (20,50))

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.74, 0.26])

        # Velas de alto contraste
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot["Open"],
                high=df_plot["High"],
                low=df_plot["Low"],
                close=df_plot["Close"],
                name="OHLC",
                increasing_line_color="rgb(16,130,59)",
                increasing_fillcolor="rgba(16,130,59,0.9)",
                decreasing_line_color="rgb(200,30,30)",
                decreasing_fillcolor="rgba(200,30,30,0.9)",
                line=dict(width=1.25),
                whiskerwidth=0.3,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        if show_sma:
            if "SMA20" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index, y=df_plot["SMA20"], mode="lines", name="SMA 20",
                        line=dict(width=1.5, dash="solid"),
                        hovertemplate="SMA20: %{y:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
            if "SMA50" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index, y=df_plot["SMA50"], mode="lines", name="SMA 50",
                        line=dict(width=1.5, dash="dot"),
                        hovertemplate="SMA50: %{y:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )

        if "Volume" in df_plot.columns:
            fig.add_trace(
                go.Bar(
                    x=df_plot.index, y=df_plot["Volume"],
                    name="Volumen", marker_color=volume_colors(df_plot),
                    opacity=0.8, hovertemplate="Volumen: %{y:,}<extra></extra>"
                ),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Volumen", row=2, col=1)

        fig.update_layout(
            title=f"{stonk} · {range_key}{suffix}",
            template="plotly_white",
            height=760,
            margin=dict(l=40, r=25, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="year", stepmode="todate", label="YTD"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
            yaxis=dict(title="Precio", showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
            dragmode="pan",
        )

        fig.update_xaxes(showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)")
        fig.update_yaxes(showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)")

        st.plotly_chart(fig, use_container_width=True)



