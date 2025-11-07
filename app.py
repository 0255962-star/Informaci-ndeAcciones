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

# ================== CONFIG ==================
st.set_page_config(
    page_title="APP Finanzas – Portafolio Activo",
    page_icon="💼",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ================== SECRETS ==================
SHEET_ID = st.secrets.get("SHEET_ID") or st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
GCP_SA = st.secrets.get("gcp_service_account", {})
SHOW_DEBUG = bool(st.secrets.get("SHOW_DEBUG", False))

if not SHEET_ID or not GCP_SA:
    st.error("Faltan secretos: `SHEET_ID` y/o credenciales de `gcp_service_account`.")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(GCP_SA, scopes=SCOPES)
gc = gspread.authorize(credentials)

# ================== SHEETS HELPERS ==================
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
    def safe_read(name, cols):
        try:
            df = read_sheet(name)
            if df.empty:
                return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df
        except Exception:
            return pd.DataFrame(columns=cols)

    tx = safe_read("Transactions", ["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"])
    settings = safe_read("Settings", ["Key","Value","Description"])
    watchlist = safe_read("Watchlist", ["Ticker","TargetWeight","Notes"])
    return tx, settings, watchlist

def get_setting(settings_df, key, default=None, cast=float):
    try:
        row = settings_df.loc[settings_df["Key"] == key, "Value"]
        if row.empty: return default
        val = row.values[0]
        return cast(val) if cast else val
    except Exception:
        return default

# ================== YFINANCE WRAPPERS ==================
def _flatten_close(df_or_panel, tickers_expected):
    """Devuelve DataFrame de Close con columnas simples."""
    if df_or_panel is None or len(df_or_panel) == 0:
        return pd.DataFrame()
    close = df_or_panel["Close"] if "Close" in df_or_panel else df_or_panel
    if isinstance(close, pd.Series):
        # un solo ticker
        name = tickers_expected[0] if tickers_expected else close.name
        close = close.to_frame(name=name)
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]
    # filtra sólo los esperados (cuando vienen extras del multipanel)
    if tickers_expected:
        close = close.loc[:, [c for c in close.columns if c in tickers_expected]]
    return close

@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices_bulk(tickers, start, end, interval="1d"):
    """Intento bulk; puede devolver vacío si yfinance falla."""
    try:
        data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        return _flatten_close(data, tickers).dropna(how="all")
    except Exception:
        return pd.DataFrame()

def fetch_prices_safe(tickers, start=None, end=None, interval="1d"):
    """
    Descarga robusta:
      1) intenta bulk
      2) si faltan columnas o viene vacío, baja ticker por ticker
    Devuelve (prices_df, failed_list)
    """
    if not tickers: return pd.DataFrame(), []
    uniq = sorted({t for t in tickers if isinstance(t, str) and t.strip()})
    if not uniq: return pd.DataFrame(), []

    if start is None:
        start = (datetime.utcnow() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    df_bulk = fetch_prices_bulk(uniq, start, end, interval)
    have_cols = set(df_bulk.columns) if not df_bulk.empty else set()

    if df_bulk.empty or len(have_cols) < len(uniq):
        # fallback ticker por ticker
        pieces = []
        failed = []
        for t in uniq:
            try:
                d = yf.download(t, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
                d = _flatten_close(d, [t]).dropna(how="all")
                if not d.empty:
                    pieces.append(d.rename(columns={d.columns[0]: t}))
                else:
                    failed.append(t)
            except Exception:
                failed.append(t)
        if pieces:
            df = pd.concat(pieces, axis=1).sort_index()
            return df, failed
        return pd.DataFrame(), uniq  # todos fallaron
    else:
        # Algunos pueden haber faltado también en bulk
        failed = [t for t in uniq if t not in df_bulk.columns]
        return df_bulk.sort_index(), failed

def last_prices_safe(tickers):
    """Último precio robusto con fallback individual. Devuelve (dict_ok, failed_list)."""
    uniq = sorted({t for t in tickers if isinstance(t, str) and t.strip()})
    if not uniq: return {}, []
    ok = {}
    failed = []
    try:
        bulk = yf.download(uniq, period="5d", interval="1d", auto_adjust=True, progress=False)
        close = _flatten_close(bulk, uniq).ffill().dropna(how="all")
        if not close.empty:
            row = close.tail(1)
            for c in row.columns:
                val = row.iloc[0][c]
                if pd.notna(val):
                    ok[c] = float(val)
    except Exception:
        pass
    # faltantes, intentar individual
    for t in uniq:
        if t in ok: continue
        try:
            d = yf.download(t, period="5d", interval="1d", auto_adjust=True, progress=False)
            d = _flatten_close(d, [t]).ffill().dropna(how="all")
            if not d.empty:
                ok[t] = float(d.iloc[-1, 0])
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    return ok, failed

# ================== METRICS ==================
def annualize_return(daily_ret: pd.Series, freq=252):
    if daily_ret.empty: return np.nan
    return float((1 + daily_ret).prod() ** (freq / max(len(daily_ret),1)) - 1)

def annualize_vol(daily_ret: pd.Series, freq=252):
    if daily_ret.empty: return np.nan
    return float(daily_ret.std(ddof=0) * np.sqrt(freq))

def sharpe(daily_ret: pd.Series, rf=0.0, freq=252):
    if daily_ret.empty: return np.nan
    er = annualize_return(daily_ret, freq)
    ev = annualize_vol(daily_ret, freq)
    return (er - rf) / ev if ev and ev > 0 else np.nan

def sortino(daily_ret: pd.Series, rf=0.0, freq=252):
    if daily_ret.empty: return np.nan
    d = daily_ret.copy(); d[d>0]=0
    dd = np.sqrt((d**2).mean()) * np.sqrt(freq)
    er = annualize_return(daily_ret, freq)
    return (er - rf) / dd if dd and dd > 0 else np.nan

def max_drawdown(cum: pd.Series):
    if cum.empty: return np.nan
    return float((cum / cum.cummax() - 1).min())

def calmar(daily_ret: pd.Series, freq=252):
    if daily_ret.empty: return np.nan
    er = annualize_return(daily_ret, freq)
    mdd = abs(max_drawdown((1+daily_ret).cumprod()))
    return er / mdd if mdd and mdd>0 else np.nan

def tracking_error(port_ret, bench_ret, freq=252):
    r = (port_ret - bench_ret).dropna()
    if r.empty: return np.nan
    return float(r.std(ddof=0) * np.sqrt(freq))

def information_ratio(port_ret, bench_ret, rf=0.0, freq=252):
    te = tracking_error(port_ret, bench_ret, freq)
    if not te or np.isnan(te): return np.nan
    return (annualize_return(port_ret,freq) - annualize_return(bench_ret,freq)) / te

def regression_beta_alpha(port_ret, bench_ret, freq=252):
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.empty: return np.nan, np.nan
    X = df.iloc[:,1].values; Y = df.iloc[:,0].values
    X_ = np.vstack([np.ones_like(X), X]).T
    a, b = np.linalg.lstsq(X_, Y, rcond=None)[0]
    alpha_ann = (1 + a) ** freq - 1
    return float(b), float(alpha_ann)

# ================== PORTFOLIO LOGIC ==================
def tidy_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    if tx.empty: return tx
    df = tx.copy()
    for c in ["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"]:
        if c not in df.columns:
            df[c] = np.nan
    # limpia tickers
    df["Ticker"] = (
        df["Ticker"].astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    df["Ticker"].replace({"": np.nan}, inplace=True)
    df = df.dropna(subset=["Ticker"])
    df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce").dt.date

    def signed(row):
        side = str(row.get("Side","")).strip().lower()
        q = float(row.get("Shares",0) or 0)
        if side in ("sell","venta","vender","-1"): return -abs(q)
        if side in ("buy","compra","1"): return  abs(q)
        return q
    df["SignedShares"] = df.apply(signed, axis=1)
    return df

def positions_from_tx(tx: pd.DataFrame):
    if tx.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    df = tidy_transactions(tx)
    uniq = sorted(df["Ticker"].unique().tolist())
    if not uniq:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])

    px_last, failed_lp = last_prices_safe(uniq)
    pos = []
    for tkr, grp in df.groupby("Ticker"):
        shares = float(grp["SignedShares"].sum())
        if abs(shares) < 1e-12: continue
        buy_mask = grp["SignedShares"] > 0
        if buy_mask.any():
            total_sh = float(grp.loc[buy_mask,"SignedShares"].sum())
            total_cost = float((grp.loc[buy_mask,"SignedShares"] * grp.loc[buy_mask,"Price"].fillna(0)).sum() + grp.loc[buy_mask,"Fees"].fillna(0).sum())
            avg_cost = total_cost / total_sh if total_sh>0 else np.nan
        else:
            avg_cost = np.nan
        mkt_price = px_last.get(tkr, np.nan)
        mv = shares * mkt_price if not np.isnan(mkt_price) else np.nan
        invested = shares * avg_cost if not np.isnan(avg_cost) else np.nan
        pl = mv - invested if not (np.isnan(mv) or np.isnan(invested)) else np.nan
        pos.append([tkr, shares, avg_cost, invested, mkt_price, mv, pl])

    pos_df = pd.DataFrame(pos, columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    if pos_df.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    pos_df = pos_df.sort_values("MarketValue", ascending=False)
    # Anota en el caption si hubo tickers sin último precio
    if failed_lp:
        st.caption("⚠️ Sin precio reciente para: " + ", ".join(failed_lp))
    return pos_df

def weights_from_positions(pos_df: pd.DataFrame):
    if pos_df.empty: return pd.Series(dtype=float)
    total = pos_df["MarketValue"].fillna(0).sum()
    if total <= 0: return pd.Series(dtype=float)
    return (pos_df.set_index("Ticker")["MarketValue"] / total).sort_values(ascending=False)

def constant_weight_portfolio_returns(price_df: pd.DataFrame, weights: pd.Series):
    if price_df is None or price_df.empty or weights is None or len(weights)==0:
        return pd.Series(dtype=float), pd.DataFrame()
    cols = [c for c in price_df.columns if c in weights.index]
    if not cols: return pd.Series(dtype=float), pd.DataFrame()
    price_df = price_df[cols].dropna(how="all")
    rets = price_df.pct_change().dropna(how="all")
    if rets.empty: return pd.Series(dtype=float), pd.DataFrame()
    w = weights.reindex(rets.columns).fillna(0).values
    port_ret = (rets * w).sum(axis=1)
    return port_ret, rets

# ================== OPTIMIZACIÓN ==================
def markowitz_max_sharpe(mean_ret, cov, rf=0.0, bounds=None):
    n = len(mean_ret)
    if n == 0: return np.array([])
    if bounds is None: bounds = [(0.0, 1.0)] * n
    def neg_sharpe(w):
        mu = float(np.dot(w, mean_ret))
        sigma = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        if sigma == 0: return 9999.0
        return -((mu - rf) / sigma)
    cons = [{"type":"eq","fun": lambda w: np.sum(w)-1.0}]
    x0 = np.array([1.0/n]*n)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":500})
    return res.x if (hasattr(res,"success") and res.success) else x0

# ================== GEMINI (opcional) ==================
def gemini_translate(text, target_lang="es"):
    if not GEMINI_API_KEY or not text: return text
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Traduce al {target_lang} y resume en 80-120 palabras, tono claro para estudiante de finanzas:\n\n{text}"
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return text

# ================== CARGA INICIAL ==================
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df, "RF", 0.03, float)
benchmark = get_setting(settings_df, "Benchmark", "^GSPC", str)
fee_per_trade = get_setting(settings_df, "FeePerTrade", 0.0, float)
slippage_bps = get_setting(settings_df, "SlippageBps", 0, float)
reb_thr = get_setting(settings_df, "RebalanceThresholdPct", 3.0, float) / 100.0
w_min = get_setting(settings_df, "MinWeightPerAsset", 0.0, float)
w_max = get_setting(settings_df, "MaxWeightPerAsset", 0.30, float)

if SHOW_DEBUG:
    st.caption(f"[DEBUG] SHEET_OK={bool(SHEET_ID)} SA_OK={bool(GCP_SA)} Benchmark={benchmark}")

# ================== SIDEBAR ==================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Herramientas"])
window = st.sidebar.selectbox("Ventana histórica", ["1Y","3Y","5Y","Max"], index=1)
period_map = {"1Y":365, "3Y":365*3, "5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow() - timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ================== HOME ==================
if page == "Mi Portafolio":
    st.title("💼 Mi Portafolio")

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún. Registra operaciones en la hoja **Transactions**.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_safe(tickers + [benchmark], start=start_date)
    # Separar benchmark si llegó
    bench_series = pd.Series(dtype=float)
    if not prices.empty and benchmark in prices.columns:
        bench_series = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")

    if prices.empty:
        msg = "No se pudieron obtener precios históricos."
        if failed:
            msg += " Tickers con problemas: " + ", ".join(failed)
        st.warning(msg + " Verifica tickers y/o intenta más tarde.")
        st.stop()

    w = weights_from_positions(pos_df)
    if w.empty:
        st.warning("No se pudieron calcular pesos de mercado.")
        st.stop()

    port_ret, asset_rets = constant_weight_portfolio_returns(prices, w)

    # KPIs
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Rend. anualizado", f"{(annualize_return(port_ret) or 0)*100:,.2f}%")
    c2.metric("Vol. anualizada", f"{(annualize_vol(port_ret) or 0)*100:,.2f}%")
    c3.metric("Sharpe", f"{(sharpe(port_ret, rf) or 0):.2f}")
    c4.metric("Sortino", f"{(sortino(port_ret, rf) or 0):.2f}")
    cum = (1+port_ret).cumprod()
    mdd = max_drawdown(cum)
    c5.metric("Max Drawdown", f"{(mdd or 0)*100:,.2f}%")
    c6.metric("Calmar", f"{(calmar(port_ret) or 0):.2f}")

    if not bench_series.empty:
        te = tracking_error(port_ret, bench_series)
        ir = information_ratio(port_ret, bench_series, rf)
        st.caption(f"Tracking Error: **{(te or 0)*100:,.2f}%** · Information Ratio: **{(ir or 0):.2f}**")

    # Curvas
    perf_df = pd.DataFrame({"Portafolio": (1+port_ret).cumprod()})
    if not bench_series.empty:
        perf_df["Benchmark"] = (1+bench_series).cumprod().reindex(perf_df.index).ffill()
    st.plotly_chart(px.line(perf_df, title="Crecimiento de 1.0"), use_container_width=True)

    # Pie
    alloc = w.reset_index(); alloc.columns = ["Ticker","Weight"]
    st.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", title="Asignación actual"), use_container_width=True)

    # Tabla
    cols = ["Ticker","Shares","AvgCost","MarketPrice","MarketValue","UnrealizedPL"]
    st.dataframe(pos_df[cols].style.format({"AvgCost":"$,.2f","MarketPrice":"$,.2f","MarketValue":"$,.2f","UnrealizedPL":"$,.2f"}), use_container_width=True)

    # Sugerencias simples
    st.subheader("🧭 Sugerencias")
    suggestions = []
    hhi = (alloc["Weight"]**2).sum()
    if hhi > 0.15: suggestions.append("Tu concentración es elevada (HHI > 0.15). Considera añadir un ETF amplio o reducir pesos top.")
    cal = calmar(port_ret)
    if not np.isnan(cal) and cal < 0.3: suggestions.append("Tu ratio Calmar es bajo. Revisa volatilidad y drawdowns.")
    if suggestions:
        for s in suggestions: st.info(s)
    else:
        st.success("Sin alertas: asignación y riesgo razonables para el periodo seleccionado.")

# ================== OPTIMIZE & REBALANCE ==================
elif page == "Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_safe(tickers, start=start_date)
    if prices.empty:
        st.warning("No hay precios para optimizar. " + ("Fallas: " + ", ".join(failed) if failed else ""))
        st.stop()

    w_cur = weights_from_positions(pos_df)
    port_ret_cur, asset_rets = constant_weight_portfolio_returns(prices, w_cur)
    if asset_rets.empty:
        st.warning("No hay retornos suficientes para optimizar.")
        st.stop()

    mean_daily = asset_rets.mean()
    cov = asset_rets.cov()
    mu_ann = (1+mean_daily)**252 - 1

    bounds = [(w_min, w_max)] * len(tickers)
    w_opt = markowitz_max_sharpe(mu_ann.values, cov.values, rf=rf, bounds=bounds)
    w_opt = pd.Series(w_opt, index=tickers).clip(lower=w_min, upper=w_max)
    w_opt /= w_opt.sum()

    port_ret_opt, _ = constant_weight_portfolio_returns(prices, w_opt)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sharpe actual", f"{(sharpe(port_ret_cur, rf) or 0):.2f}")
    c2.metric("Sharpe propuesto", f"{(sharpe(port_ret_opt, rf) or 0):.2f}")
    c3.metric("Vol. actual", f"{(annualize_vol(port_ret_cur) or 0)*100:,.2f}%")
    c4.metric("Vol. propuesta", f"{(annualize_vol(port_ret_opt) or 0)*100:,.2f}%")

    compare = pd.DataFrame({"Weight Actual": w_cur, "Weight Propuesto": w_opt}).fillna(0)
    compare["Δ (pp)"] = (compare["Weight Propuesto"] - compare["Weight Actual"])*100
    st.dataframe(compare.style.format({"Weight Actual":"{:.2%}","Weight Propuesto":"{:.2%}","Δ (pp)":"{:.2f}"}), use_container_width=True)

    st.subheader("Órdenes sugeridas (simulación)")
    portfolio_value = pos_df["MarketValue"].sum()
    orders = []
    for t in tickers:
        px = float(pos_df.loc[pos_df["Ticker"]==t, "MarketPrice"].fillna(0).values[0])
        if px <= 0: continue
        tgt_value = portfolio_value * w_opt.get(t, 0.0)
        tgt_shares = np.floor(tgt_value / px)
        cur_shares = float(pos_df.loc[pos_df["Ticker"]==t, "Shares"].values[0])
        delta = tgt_shares - cur_shares
        if delta != 0:
            side = "Buy" if delta>0 else "Sell"
            est_cost = abs(delta)*px + fee_per_trade + (abs(delta)*px)*(slippage_bps/10000)
            orders.append([t, side, int(delta), px, est_cost])
    ord_df = pd.DataFrame(orders, columns=["Ticker","Side","Shares","EstPrice","EstCost"])
    if ord_df.empty:
        st.success("Tu portafolio ya está muy cercano al objetivo propuesto.")
    else:
        st.dataframe(ord_df.style.format({"EstPrice":"$,.2f","EstCost":"$,.2f"}), use_container_width=True)
        with st.expander("Confirmar manualmente (escribe en Transactions)"):
            if st.button("Registrar todas las órdenes sugeridas"):
                today = datetime.utcnow().date().strftime("%Y-%m-%d")
                for _, r in ord_df.iterrows():
                    write_sheet_append("Transactions", [r["Ticker"], today, r["Side"], r["Shares"], r["EstPrice"], fee_per_trade, "Auto-rebalance simulado"])
                st.success("Órdenes registradas. Pulsa R para refrescar.")

# ================== EVALUAR CANDIDATO ==================
elif page == "Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún.")
        st.stop()

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ", "")
    if not tkr:
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    all_tickers = sorted(set(tickers + [tkr]))
    prices, failed = fetch_prices_safe(all_tickers, start=start_date)
    if prices.empty:
        st.warning("No se pudieron descargar precios para el candidato. " + ("Fallas: " + ", ".join(failed) if failed else ""))
        st.stop()

    w_cur = weights_from_positions(pos_df)
    alloc_mode = st.radio("Forma de evaluación", ["Asignar porcentaje","Añadir acciones"], horizontal=True)

    last_map, failed_lp = last_prices_safe([tkr])
    last = last_map.get(tkr, np.nan)
    if np.isnan(last):
        st.warning("Ticker inválido o sin precio reciente.")
        st.stop()

    if alloc_mode == "Asignar porcentaje":
        pct = st.slider("Peso objetivo del candidato", 0.0, 0.40, 0.10, 0.01)
        w_new = (w_cur * (1 - pct)).reindex(all_tickers).fillna(0.0)
        w_new[tkr] += pct
    else:
        qty = st.number_input("Acciones a comprar (simulado)", min_value=1, value=5, step=1)
        pv = pos_df["MarketValue"].sum()
        add_value = qty * last
        base = pv + add_value if (pv + add_value) > 0 else 1.0
        w_new = (w_cur * pv) / base
        w_new = w_new.reindex(all_tickers).fillna(0.0)
        w_new[tkr] += add_value / base

    port_cur, _ = constant_weight_portfolio_returns(prices, w_cur.reindex(prices.columns, fill_value=0))
    port_new, assets_new = constant_weight_portfolio_returns(prices, w_new.reindex(prices.columns, fill_value=0))

    cols = st.columns(4)
    sh_old = sharpe(port_cur, rf); sh_new = sharpe(port_new, rf)
    cols[0].metric("Sharpe actual", f"{(sh_old or 0):.2f}")
    cols[1].metric("Sharpe con candidato", f"{(sh_new or 0):.2f}", delta=f"{((sh_new or 0)-(sh_old or 0)):.2f}")
    cum_cur = (1 + port_cur).cumprod() if not port_cur.empty else pd.Series(dtype=float)
    cum_new = (1 + port_new).cumprod() if not port_new.empty else pd.Series(dtype=float)
    mdd_cur = max_drawdown(cum_cur) if not cum_cur.empty else np.nan
    mdd_new = max_drawdown(cum_new) if not cum_new.empty else np.nan
    cols[2].metric("MDD actual", f"{(mdd_cur or 0)*100:,.2f}%")
    cols[3].metric("MDD con candidato", f"{(mdd_new or 0)*100:,.2f}%", delta=f"{(((mdd_new or 0)-(mdd_cur or 0))*100):.2f}%")

    rho = np.nan
    if tkr in assets_new.columns and not port_cur.empty:
        rho = assets_new[tkr].corr(port_cur.reindex(assets_new.index))
        if not np.isnan(rho):
            st.caption(f"Correlación candidato vs. cartera: **{rho:.2f}**")

    delta_sharpe = (sh_new or np.nan) - (sh_old or np.nan)
    rule_pass = (
        (not np.isnan(delta_sharpe) and delta_sharpe >= 0.03) and
        (not np.isnan(mdd_cur) and not np.isnan(mdd_new) and (mdd_new - mdd_cur) >= -0.02) and
        (np.isnan(rho) or rho <= 0.75)
    )
    if rule_pass:
        st.success("✅ Recomendación positiva.")
    else:
        st.warning("⚠️ La mejora no supera los umbrales definidos.")

    with st.expander("Registrar compra simulada en Transactions"):
        if st.button("Registrar (Buy)"):
            today = datetime.utcnow().date().strftime("%Y-%m-%d")
            write_sheet_append("Transactions", [tkr, today, "Buy", (5 if alloc_mode=='Añadir acciones' else 0), last, 0.0, "Evaluación aprobada"])
            st.success("Operación registrada.")

# ================== EXPLORAR / RESEARCH ==================
elif page == "Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr = st.text_input("Ticker", value="MSFT").upper().strip().replace(" ", "")
    if tkr:
        hist = yf.download(tkr, period="1y", interval="1d", auto_adjust=True, progress=False)
        if hist.empty:
            st.warning("No hay datos para ese ticker.")
            st.stop()

        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.update_layout(title=f"Velas: {tkr}", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        df = hist.copy()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        st.plotly_chart(px.line(df[["Close","SMA20","SMA50"]], title="SMAs"), use_container_width=True)

        info = {}
        try:
            info = yf.Ticker(tkr).info or {}
        except Exception:
            info = {}
        long_name = info.get("longName") or tkr
        summary = info.get("longBusinessSummary") or ""
        st.subheader(long_name)
        st.write(gemini_translate(summary, "es") if summary else "Sin descripción disponible.")

# ================== HERRAMIENTAS ==================
elif page == "Herramientas":
    st.title("🧰 Herramientas")
    tabs = st.tabs(["β / α", "CAPM rápido", "Monte Carlo"])

    with tabs[0]:
        st.subheader("β / α vs Benchmark")
        pos_df = positions_from_tx(tx_df)
        if pos_df.empty:
            st.info("Sin posiciones.")
        else:
            tickers = pos_df["Ticker"].tolist()
            prices, _ = fetch_prices_safe(tickers + [benchmark], start=start_date)
            if prices.empty or benchmark not in prices.columns:
                st.warning("No hay suficientes datos del benchmark para β/α.")
            else:
                core = [c for c in prices.columns if c != benchmark]
                w = weights_from_positions(pos_df)
                port_ret, _ = constant_weight_portfolio_returns(prices[core], w)
                bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
                if bench_ret.empty or port_ret.empty:
                    st.warning("No hay suficientes datos.")
                else:
                    beta, alpha_ann = regression_beta_alpha(port_ret, bench_ret)
                    c1,c2 = st.columns(2)
                    c1.metric("Beta", f"{(beta or 0):.2f}")
                    c2.metric("Alpha anualizado", f"{(alpha_ann or 0)*100:,.2f}%")

    with tabs[1]:
        st.subheader("CAPM rápido")
        exp_mkt = st.number_input("E[Rm] esperado (anual)", value=0.08, step=0.01, format="%.2f")
        beta_i = st.number_input("Beta del activo", value=1.00, step=0.10, format="%.2f")
        rf_capm = st.number_input("r_f (anual)", value=float(rf), step=0.005, format="%.3f")
        er = rf_capm + beta_i*(exp_mkt - rf_capm)
        st.metric("E[Ri] (CAPM)", f"{er*100:,.2f}%")

    with tabs[2]:
        st.subheader("Simulación Monte Carlo (portafolio)")
        pos_df = positions_from_tx(tx_df)
        if pos_df.empty:
            st.info("Sin posiciones.")
        else:
            tickers = pos_df["Ticker"].tolist()
            prices, _ = fetch_prices_safe(tickers, start=start_date)
            if prices.empty:
                st.warning("No hay precios para simular.")
            else:
                w = weights_from_positions(pos_df)
                port_ret, asset_rets = constant_weight_portfolio_returns(prices, w)
                if port_ret.empty:
                    st.warning("No hay retornos suficientes.")
                else:
                    mu = port_ret.mean(); sigma = port_ret.std(ddof=0)
                    horizon_days = st.slider("Horizonte (días)", 30, 365, 180, 10)
                    sims = st.slider("N simulaciones", 200, 5000, 1000, 100)
                    rng = np.random.default_rng(42)
                    sim_end = []
                    for _ in range(sims):
                        path = rng.normal(mu, sigma, size=horizon_days)
                        sim_end.append(np.prod(1+path))
                    sim_end = np.array(sim_end)
                    p5, p50, p95 = np.percentile(sim_end, [5,50,95])
                    c1,c2,c3 = st.columns(3)
                    c1.metric("P5", f"{(p5-1)*100:,.2f}%")
                    c2.metric("P50", f"{(p50-1)*100:,.2f}%")
                    c3.metric("P95", f"{(p95-1)*100:,.2f}%")
                    st.plotly_chart(px.histogram(sim_end-1, nbins=40, title="Distribución de rendimientos simulados"), use_container_width=True)
