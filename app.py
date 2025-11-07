
---

# `app.py`
```python
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

import gspread
from google.oauth2.service_account import Credentials

# ============ CONFIG ============
st.set_page_config(
    page_title="APP Finanzas – Portafolio Activo",
    page_icon="💼",
    layout="wide",
)

# Pequeño estilo adicional
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 1rem; }
.css-1r6slb0 { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============ SECRETS ============
SHEET_ID = st.secrets.get("SHEET_ID", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GCP_SA = st.secrets.get("gcp_service_account", {})

if not SHEET_ID or not GCP_SA:
    st.error("Faltan secretos: `SHEET_ID` y/o credenciales de `gcp_service_account`.")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
credentials = Credentials.from_service_account_info(GCP_SA, scopes=SCOPES)
gc = gspread.authorize(credentials)

# ============ UTILS: SHEETS ============
@st.cache_data(show_spinner=False, ttl=600)
def read_sheet(sheet_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    return df

def write_sheet_append(sheet_name: str, row_values: list):
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(sheet_name)
    ws.append_row(row_values, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False, ttl=600)
def load_all_data():
    try:
        tx = read_sheet("Transactions")
    except Exception:
        tx = pd.DataFrame(columns=["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"])
    try:
        settings = read_sheet("Settings")
    except Exception:
        settings = pd.DataFrame(columns=["Key","Value","Description"])
    try:
        watchlist = read_sheet("Watchlist")
    except Exception:
        watchlist = pd.DataFrame(columns=["Ticker","TargetWeight","Notes"])
    return tx, settings, watchlist

def get_setting(settings_df, key, default=None, cast=float):
    try:
        row = settings_df.loc[settings_df["Key"] == key, "Value"]
        if row.empty:
            return default
        val = row.values[0]
        if cast is None:
            return val
        return cast(val)
    except Exception:
        return default

# ============ DATA: PRICES ============
@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices(tickers, start=None, end=None, interval="1d"):
    if not tickers:
        return pd.DataFrame()
    if start is None:
        start = (datetime.utcnow() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    return data

@st.cache_data(show_spinner=False, ttl=900)
def last_prices(tickers):
    if not tickers:
        return {}
    info = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(info, pd.Series):
        info = info.to_frame()
    last = info.ffill().iloc[-1].to_dict()
    return last

# ============ METRICS ============
def annualize_return(daily_ret: pd.Series, freq=252):
    m = (1 + daily_ret).prod() ** (freq / len(daily_ret)) - 1
    return float(m)

def annualize_vol(daily_ret: pd.Series, freq=252):
    return float(daily_ret.std(ddof=0) * np.sqrt(freq))

def sharpe(daily_ret: pd.Series, rf=0.0, freq=252):
    er = annualize_return(daily_ret, freq)
    ev = annualize_vol(daily_ret, freq)
    return (er - rf) / ev if ev > 0 else np.nan

def sortino(daily_ret: pd.Series, rf=0.0, freq=252):
    downside = daily_ret.copy()
    downside[downside > 0] = 0
    dd = np.sqrt((downside**2).mean()) * np.sqrt(freq)
    er = annualize_return(daily_ret, freq)
    return (er - rf) / dd if dd > 0 else np.nan

def max_drawdown(cum: pd.Series):
    peak = cum.cummax()
    dd = (cum/peak - 1).min()
    return float(dd)

def calmar(daily_ret: pd.Series, freq=252):
    er = annualize_return(daily_ret, freq)
    cum = (1 + daily_ret).cumprod()
    mdd = abs(max_drawdown(cum))
    return er / mdd if mdd > 0 else np.nan

def tracking_error(port_ret, bench_ret, freq=252):
    rets = (port_ret - bench_ret).dropna()
    return float(rets.std(ddof=0) * np.sqrt(freq))

def information_ratio(port_ret, bench_ret, rf=0.0, freq=252):
    te = tracking_error(port_ret, bench_ret, freq)
    if te == 0 or np.isnan(te):
        return np.nan
    er_p = annualize_return(port_ret, freq)
    er_b = annualize_return(bench_ret, freq)
    return (er_p - er_b) / te

def regression_beta_alpha(port_ret, bench_ret, freq=252):
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan
    X = df.iloc[:,1].values
    Y = df.iloc[:,0].values
    X_ = np.vstack([np.ones_like(X), X]).T
    coef = np.linalg.lstsq(X_, Y, rcond=None)[0]
    alpha_daily, beta = coef[0], coef[1]
    alpha_ann = (1 + alpha_daily) ** freq - 1
    return float(beta), float(alpha_ann)

# ============ PORTFOLIO LOGIC ============
def tidy_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    if tx.empty:
        return tx
    df = tx.copy()
    for c in ["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce").dt.date
    # Soporta modelo con Side o con signo en Shares
    def signed(row):
        side = str(row.get("Side","")).strip().lower()
        q = float(row.get("Shares",0) or 0)
        if side in ("sell","venta","vender","-1"):
            return -abs(q)
        if side in ("buy","compra","1"):
            return abs(q)
        return q  # si no hay side, respeta el signo de Shares
    df["SignedShares"] = df.apply(signed, axis=1)
    return df

def positions_from_tx(tx: pd.DataFrame):
    if tx.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    df = tidy_transactions(tx)
    px_last = last_prices(sorted(df["Ticker"].unique()))
    # costo promedio
    pos = []
    for tkr, grp in df.groupby("Ticker"):
        shares = grp["SignedShares"].sum()
        if abs(shares) < 1e-12:
            continue
        # avg cost solo con compras (>0)
        buy_mask = grp["SignedShares"] > 0
        if buy_mask.any():
            total_shares = grp.loc[buy_mask,"SignedShares"].sum()
            total_cost = (grp.loc[buy_mask,"SignedShares"] * grp.loc[buy_mask,"Price"].fillna(0)).sum() + grp.loc[buy_mask,"Fees"].fillna(0).sum()
            avg_cost = total_cost / total_shares if total_shares > 0 else np.nan
        else:
            avg_cost = np.nan
        mkt_price = px_last.get(tkr, np.nan)
        mv = shares * mkt_price if not np.isnan(mkt_price) else np.nan
        invested = shares * avg_cost if not np.isnan(avg_cost) else np.nan
        pl = mv - invested if not (np.isnan(mv) or np.isnan(invested)) else np.nan
        pos.append([tkr, shares, avg_cost, invested, mkt_price, mv, pl])
    pos_df = pd.DataFrame(pos, columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    pos_df = pos_df.sort_values("MarketValue", ascending=False)
    return pos_df

def weights_from_positions(pos_df: pd.DataFrame):
    if pos_df.empty:
        return pd.Series(dtype=float)
    total = pos_df["MarketValue"].fillna(0).sum()
    if total <= 0:
        return pd.Series(dtype=float)
    w = (pos_df.set_index("Ticker")["MarketValue"] / total).sort_values(ascending=False)
    return w

def constant_weight_portfolio_returns(price_df: pd.DataFrame, weights: pd.Series):
    price_df = price_df[weights.index].dropna()
    rets = price_df.pct_change().dropna()
    w = weights.reindex(price_df.columns).fillna(0).values
    port_ret = (rets * w).sum(axis=1)
    return port_ret, rets

# ======== OPTIMIZATION (Max Sharpe) ========
def markowitz_max_sharpe(mean_ret, cov, rf=0.0, bounds=None):
    n = len(mean_ret)
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    def neg_sharpe(w):
        mu = np.dot(w, mean_ret)
        sigma = np.sqrt(np.dot(w, np.dot(cov, w)))
        if sigma == 0:
            return 9999
        return -( (mu - rf) / sigma )
    cons = [{"type":"eq","fun": lambda w: np.sum(w)-1.0}]
    x0 = np.array([1.0/n]*n)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":500})
    return res.x if res.success else x0

# ======== GEMINI (opcional) ========
def gemini_translate(text, target_lang="es"):
    if not GEMINI_API_KEY or not text:
        return text
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Traduce al {target_lang} y resume en 80-120 palabras, tono claro para estudiante de finanzas:\n\n{text}"
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return text

# ============ SIDEBAR ============
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df, "RF", 0.03, float)
benchmark = get_setting(settings_df, "Benchmark", "^GSPC", str)
fee_per_trade = get_setting(settings_df, "FeePerTrade", 0.0, float)
slippage_bps = get_setting(settings_df, "SlippageBps", 0, float)
reb_thr = get_setting(settings_df, "RebalanceThresholdPct", 3.0, float) / 100.0
w_min = get_setting(settings_df, "MinWeightPerAsset", 0.0, float)
w_max = get_setting(settings_df, "MaxWeightPerAsset", 0.30, float)

st.sidebar.title("📊 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["Mi Portafolio", "Optimizar y Rebalancear", "Evaluar Candidato", "Explorar / Research", "Herramientas"],
)

window = st.sidebar.selectbox("Ventana histórica", ["1Y","3Y","5Y","Max"], index=1)
period_map = {"1Y":365, "3Y":365*3, "5Y":365*5}
if window == "Max":
    start_date = None
else:
    start_date = (datetime.utcnow() - timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ============ HOME ============
if page == "Mi Portafolio":
    st.title("💼 Mi Portafolio")

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún. Registra operaciones en la hoja **Transactions**.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices = fetch_prices(tickers + [benchmark], start=start_date)
    prices = prices.dropna(how="all")
    last_px = last_prices(tickers)

    w = weights_from_positions(pos_df)
    if w.empty:
        st.warning("No se pudieron calcular pesos de mercado.")
        st.stop()

    # Rendimientos
    bench_series = prices[[benchmark]].dropna().pct_change().dropna()[benchmark] if benchmark in prices.columns else pd.Series(dtype=float)
    port_ret, asset_rets = constant_weight_portfolio_returns(prices, w)

    # KPIs
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    col1.metric("Rend. anualizado", f"{annualize_return(port_ret)*100:,.2f}%")
    col2.metric("Vol. anualizada", f"{annualize_vol(port_ret)*100:,.2f}%")
    col3.metric("Sharpe", f"{sharpe(port_ret, rf):.2f}")
    col4.metric("Sortino", f"{sortino(port_ret, rf):.2f}")
    cum = (1+port_ret).cumprod()
    mdd = max_drawdown(cum)
    col5.metric("Max Drawdown", f"{mdd*100:,.2f}%")
    col6.metric("Calmar", f"{calmar(port_ret):.2f}")

    # vs Benchmark
    if not bench_series.empty:
        te = tracking_error(port_ret, bench_series)
        ir = information_ratio(port_ret, bench_series, rf)
        st.caption(f"Tracking Error: **{te*100:,.2f}%** · Information Ratio: **{ir:.2f}**")

    # Gráfico de crecimiento
    perf_df = pd.DataFrame({
        "Portafolio": (1+port_ret).cumprod()
    })
    if not bench_series.empty:
        perf_df["Benchmark"] = (1+bench_series).cumprod().reindex(perf_df.index).ffill()
    fig = px.line(perf_df, title="Crecimiento de 1.0")
    st.plotly_chart(fig, use_container_width=True)

    # Asignación
    alloc = w.reset_index()
    alloc.columns = ["Ticker","Weight"]
    alloc["Weight(%)"] = alloc["Weight"]*100
    fig_alloc = px.pie(alloc, names="Ticker", values="Weight", title="Asignación actual")
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Tabla posiciones
    show_cols = ["Ticker","Shares","AvgCost","MarketPrice","MarketValue","UnrealizedPL"]
    st.dataframe(pos_df[show_cols].style.format({"AvgCost":"$,.2f","MarketPrice":"$,.2f","MarketValue":"$,.2f","UnrealizedPL":"$,.2f"}), use_container_width=True)

    # Sugerencias (simples)
    st.subheader("🧭 Sugerencias")
    suggestions = []
    # Diversificación (HHI simple)
    hhi = (alloc["Weight"]**2).sum()
    if hhi > 0.15:
        suggestions.append("Tu concentración es elevada (HHI > 0.15). Considera añadir un ETF amplio o reducir los pesos top.")
    if not bench_series.empty and calmar(port_ret) < 0.3:
        suggestions.append("Tu ratio Calmar es bajo. Revisa la volatilidad o drawdowns recientes.")
    if suggestions:
        for s in suggestions:
            st.info(s)
    else:
        st.success("Sin alertas: asignación y riesgo razonables para el periodo seleccionado.")

# ============ OPTIMIZE & REBALANCE ============
elif page == "Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices = fetch_prices(tickers, start=start_date)
    prices = prices.dropna(how="all")
    port_ret, asset_rets = constant_weight_portfolio_returns(prices, weights_from_positions(pos_df))
    mean_daily = asset_rets.mean()
    cov = asset_rets.cov()
    mu_ann = (1+mean_daily)**252 - 1

    bounds = [(w_min, w_max)] * len(tickers)
    w_opt = markowitz_max_sharpe(mu_ann.values, cov.values, rf=rf, bounds=bounds)
    w_opt = pd.Series(w_opt, index=tickers).clip(lower=w_min, upper=w_max)
    w_opt /= w_opt.sum()

    w_cur = weights_from_positions(pos_df)
    # Impacto
    port_ret_cur, _ = constant_weight_portfolio_returns(prices, w_cur)
    port_ret_opt, _ = constant_weight_portfolio_returns(prices, w_opt)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sharpe actual", f"{sharpe(port_ret_cur, rf):.2f}")
    c2.metric("Sharpe propuesto", f"{sharpe(port_ret_opt, rf):.2f}")
    c3.metric("Vol. actual", f"{annualize_vol(port_ret_cur)*100:,.2f}%")
    c4.metric("Vol. propuesta", f"{annualize_vol(port_ret_opt)*100:,.2f}%")

    # Tabla pesos
    compare = pd.DataFrame({
        "Weight Actual": w_cur,
        "Weight Propuesto": w_opt
    }).fillna(0)
    compare["Δ (pp)"] = (compare["Weight Propuesto"] - compare["Weight Actual"])*100
    st.dataframe(compare.style.format({"Weight Actual":"{:.2%}","Weight Propuesto":"{:.2%}","Δ (pp)":"{:.2f}"}), use_container_width=True)

    # Órdenes sugeridas
    st.subheader("Órdenes sugeridas (simulación)")
    portfolio_value = pos_df["MarketValue"].sum()
    target_shares = []
    for t in tickers:
        px = pos_df.loc[pos_df["Ticker"]==t, "MarketPrice"].values[0]
        tgt_value = portfolio_value * w_opt[t]
        tgt_shares = np.floor(tgt_value / px)  # entero
        cur_shares = pos_df.loc[pos_df["Ticker"]==t, "Shares"].values[0]
        delta = tgt_shares - cur_shares
        if delta != 0:
            cost = abs(delta)*px + fee_per_trade + (abs(delta)*px)*(slippage_bps/10000)
            side = "Buy" if delta>0 else "Sell"
            target_shares.append([t, side, int(delta), px, cost])
    ord_df = pd.DataFrame(target_shares, columns=["Ticker","Side","Shares","EstPrice","EstCost"])
    if ord_df.empty:
        st.success("Tu portafolio ya está muy cercano al objetivo propuesto.")
    else:
        st.dataframe(ord_df.style.format({"EstPrice":"$,.2f","EstCost":"$,.2f"}), use_container_width=True)
        st.caption("Estas órdenes **no** se registran automáticamente. Si confirmas manualmente, se escriben en la hoja *Transactions*.")

        with st.expander("Confirmar manualmente (escribe en Transactions)"):
            if st.button("Registrar todas las órdenes sugeridas"):
                today = datetime.utcnow().date().strftime("%Y-%m-%d")
                for _, r in ord_df.iterrows():
                    write_sheet_append("Transactions", [r["Ticker"], today, r["Side"], r["Shares"], r["EstPrice"], fee_per_trade, "Auto-rebalance simulado"])
                st.success("Órdenes registradas en *Transactions*. Pulsa 'R' para refrescar datos.")

# ============ EVALUAR CANDIDATO ============
elif page == "Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices_all = fetch_prices(tickers, start=start_date)
    w_cur = weights_from_positions(pos_df)

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip()
    alloc_mode = st.radio("Forma de evaluación", ["Asignar porcentaje","Añadir acciones"], horizontal=True)

    if alloc_mode == "Asignar porcentaje":
        pct = st.slider("Peso objetivo del candidato", 0.0, 0.40, 0.10, 0.01)
        # Nuevo set de pesos: asigna pct al candidato y escala los demás proporcionalmente
        all_tickers = sorted(set(tickers + [tkr]))
        prices = fetch_prices(all_tickers, start=start_date)
        last = last_prices([tkr]).get(tkr, np.nan)
        if np.isnan(last):
            st.warning("Ticker inválido o sin datos.")
            st.stop()
        w_new = (w_cur * (1 - pct)).reindex(all_tickers).fillna(0.0)
        w_new[tkr] += pct

    else:
        qty = st.number_input("Acciones a comprar (simulado)", min_value=1, value=5, step=1)
        all_tickers = sorted(set(tickers + [tkr]))
        prices = fetch_prices(all_tickers, start=start_date)
        last = last_prices([tkr]).get(tkr, np.nan)
        if np.isnan(last):
            st.warning("Ticker inválido o sin datos.")
            st.stop()
        # Recalcula pesos asumiendo compra financiada con efectivo externo
        pv = pos_df["MarketValue"].sum()
        add_value = qty * last
        w_new = (w_cur * pv) / (pv + add_value)
        w_new = w_new.reindex(all_tickers).fillna(0.0)
        w_new[tkr] += add_value / (pv + add_value)

    # Métricas Δ
    port_cur, assets_cur = constant_weight_portfolio_returns(prices, w_cur.reindex(prices.columns).fillna(0))
    port_new, assets_new = constant_weight_portfolio_returns(prices, w_new.reindex(prices.columns).fillna(0))
    rf_local = rf

    cols = st.columns(4)
    cols[0].metric("Sharpe actual", f"{sharpe(port_cur, rf_local):.2f}")
    cols[1].metric("Sharpe con candidato", f"{sharpe(port_new, rf_local):.2f}", delta=f"{(sharpe(port_new, rf_local)-sharpe(port_cur, rf_local)):.2f}")
    cum_cur = (1 + port_cur).cumprod()
    cum_new = (1 + port_new).cumprod()
    mdd_cur = max_drawdown(cum_cur)
    mdd_new = max_drawdown(cum_new)
    cols[2].metric("MDD actual", f"{mdd_cur*100:,.2f}%")
    cols[3].metric("MDD con candidato", f"{mdd_new*100:,.2f}%", delta=f"{(mdd_new-mdd_cur)*100:,.2f}%")

    # Correlación candidato vs cartera (aprox)
    if tkr in assets_new.columns:
        rho = assets_new[tkr].corr(port_cur.reindex(assets_new.index))
        st.caption(f"Correlación candidato vs. cartera: **{rho:.2f}** (menor es mejor para diversificar).")

    # Semáforo
    delta_sharpe = sharpe(port_new, rf_local) - sharpe(port_cur, rf_local)
    rule_pass = (delta_sharpe >= 0.03) and ((mdd_new - mdd_cur) >= -0.02) and (rho <= 0.75 if 'rho' in locals() and not np.isnan(rho) else True)
    if rule_pass:
        st.success("✅ Recomendación positiva: mejora Sharpe y el riesgo es aceptable bajo reglas actuales.")
    else:
        st.warning("⚠️ Revisa: la mejora no supera los umbrales definidos.")

    with st.expander("Registrar compra simulada en Transactions"):
        if st.button("Registrar (Buy)"):
            today = datetime.utcnow().date().strftime("%Y-%m-%d")
            write_sheet_append("Transactions", [tkr, today, "Buy", qty if alloc_mode=="Añadir acciones" else 0, last, 0.0, "Evaluación aprobada"])
            st.success("Operación registrada.")

# ============ EXPLORAR / RESEARCH ============
elif page == "Explorar / Research":
    st.title("🔎 Explorar / Research")

    tkr = st.text_input("Ticker", value="MSFT").upper().strip()
    if tkr:
        hist = yf.download(tkr, period="1y", interval="1d", auto_adjust=True, progress=False)
        if hist.empty:
            st.warning("No hay datos para ese ticker.")
            st.stop()

        # Velas
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
        )])
        fig.update_layout(title=f"Velas: {tkr}", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # SMAs
        df = hist.copy()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        sma = px.line(df[["Close","SMA20","SMA50"]], title="SMAs")
        st.plotly_chart(sma, use_container_width=True)

        # Descripción / Traducción
        info = yf.Ticker(tkr).info or {}
        long_name = info.get("longName") or tkr
        summary = info.get("longBusinessSummary") or ""
        st.subheader(long_name)
        if summary:
            st.write(gemini_translate(summary, "es") if GEMINI_API_KEY else summary)
        else:
            st.caption("Sin descripción disponible.")

# ============ HERRAMIENTAS ============
elif page == "Herramientas":
    st.title("🧰 Herramientas")

    sub = st.tabs(["β / α", "CAPM rápido", "Monte Carlo"])

    # --- Beta / Alpha ---
    with sub[0]:
        st.subheader("β / α vs Benchmark")
        pos_df = positions_from_tx(tx_df)
        if pos_df.empty:
            st.info("Sin posiciones.")
        else:
            tickers = pos_df["Ticker"].tolist()
            prices = fetch_prices(tickers + [benchmark], start=start_date)
            w = weights_from_positions(pos_df)
            port_ret, _ = constant_weight_portfolio_returns(prices, w)
            bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark] if benchmark in prices.columns else pd.Series(dtype=float)
            if bench_ret.empty or port_ret.empty:
                st.warning("No hay suficientes datos.")
            else:
                beta, alpha_ann = regression_beta_alpha(port_ret, bench_ret)
                c1,c2 = st.columns(2)
                c1.metric("Beta", f"{beta:.2f}")
                c2.metric("Alpha anualizado", f"{alpha_ann*100:,.2f}%")

    # --- CAPM ---
    with sub[1]:
        st.subheader("CAPM rápido")
        exp_mkt = st.number_input("E[Rm] esperado (anual)", value=0.08, step=0.01, format="%.2f")
        beta_i = st.number_input("Beta del activo", value=1.00, step=0.10, format="%.2f")
        rf_capm = st.number_input("r_f (anual)", value=rf, step=0.005, format="%.3f")
        er = rf_capm + beta_i*(exp_mkt - rf_capm)
        st.metric("E[Ri] (CAPM)", f"{er*100:,.2f}%")

    # --- Monte Carlo ---
    with sub[2]:
        st.subheader("Simulación Monte Carlo (portafolio)")
        pos_df = positions_from_tx(tx_df)
        if pos_df.empty:
            st.info("Sin posiciones.")
        else:
            tickers = pos_df["Ticker"].tolist()
            prices = fetch_prices(tickers, start=start_date)
            w = weights_from_positions(pos_df)
            port_ret, asset_rets = constant_weight_portfolio_returns(prices, w)
            mu = port_ret.mean()
            sigma = port_ret.std(ddof=0)
            horizon_days = st.slider("Horizonte (días)", 30, 365, 180, 10)
            sims = st.slider("N simulaciones", 200, 5000, 1000, 100)
            rng = np.random.default_rng(42)
            sim_end = []
            for _ in range(sims):
                path = rng.normal(mu, sigma, size=horizon_days)
                sim_end.append(np.prod(1+path))
            sim_end = np.array(sim_end)
            p5, p50, p95 = np.percentile(sim_end, [5,50,95])
            st.metric("P5", f"{(p5-1)*100:,.2f}%"); st.metric("P50", f"{(p50-1)*100:,.2f}%"); st.metric("P95", f"{(p95-1)*100:,.2f}%")
            fig = px.histogram(sim_end-1, nbins=40, title="Distribución de rendimientos simulados")
            st.plotly_chart(fig, use_container_width=True)

