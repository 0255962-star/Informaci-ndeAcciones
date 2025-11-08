# app.py
# APP Finanzas – Portafolio Activo (versión robusta con fallback Stooq .US y proxy SPY para ^GSPC)
# - Tabla estilo broker: Avg Buy, Last, P/L $, P/L% compra, Δ% ventana, Peso, Valor
# - Capa de precios resiliente (Yahoo batch -> Yahoo per-ticker -> Stooq con mapeo .US y proxy benchmark)
# - Optimización continúa con los tickers que sí tengan datos (excluye fallidos)
# - Conexión a Google Sheets (Transactions / Settings / Watchlist)

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

import gspread
from google.oauth2.service_account import Credentials

# ============== LOOK & FEEL ==============
st.set_page_config(page_title="APP Finanzas – Portafolio Activo", page_icon="💼", layout="wide")
st.markdown("""
<style>
:root {
  --bg: #0b0f1a;
  --card: #141927;
  --muted: #8b93a7;
  --brand: #7c6cff;
  --up: #18b26b;
  --down: #ff4d4f;
}
.block-container { padding-top: 0.8rem; }
div[data-testid="stMetricValue"]{font-size:1.6rem}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""", unsafe_allow_html=True)

# ============== SECRETS / GSHEETS ==============
SHEET_ID = st.secrets.get("SHEET_ID") or st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
GCP_SA = st.secrets.get("gcp_service_account", {})

if not SHEET_ID or not GCP_SA:
    st.error("Faltan secretos: `SHEET_ID` y/o `gcp_service_account`.")
    st.stop()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(GCP_SA, scopes=SCOPES)
gc = gspread.authorize(credentials)

# ============== CACHE HELPERS ==============
def refresh_data():
    try: read_sheet.clear()
    except Exception: pass
    try: load_all_data.clear()
    except Exception: pass
    try: st.cache_data.clear()
    except Exception: pass

@st.cache_data(show_spinner=False, ttl=600)
def read_sheet(name:str)->pd.DataFrame:
    sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet(name)
    return pd.DataFrame(ws.get_all_records())

# ============== LOAD SHEETS (tolerante a columnas) ==============
@st.cache_data(show_spinner=False, ttl=600)
def load_all_data():
    def safe(name, cols):
        try:
            df = read_sheet(name)
            if df.empty: return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns: df[c]=np.nan
            return df
        except Exception:
            return pd.DataFrame(columns=cols)

    tx_cols = ["TradeID","Account","Ticker","Name","AssetType","Currency",
               "TradeDate","Side","Shares","Price","Fees","Taxes","FXRate",
               "GrossAmount","NetAmount","LotID","Source","Notes"]
    tx = safe("Transactions", tx_cols)
    settings = safe("Settings", ["Key","Value","Description"])
    watchlist = safe("Watchlist", ["Ticker","TargetWeight","Notes"])
    return tx, settings, watchlist

def get_setting(settings_df, key, default=None, cast=float):
    try:
        s = settings_df.loc[settings_df["Key"]==key,"Value"]
        if s.empty: return default
        return cast(s.values[0]) if cast else s.values[0]
    except Exception:
        return default

# ============== CAPA DE PRECIOS ROBUSTA (con mapeo .US y proxy SPY) =========================
def _clean_tickers(tickers):
    uniq = []
    for t in tickers:
        if not isinstance(t,str): continue
        t2 = t.upper().strip().replace(" ", "")
        if t2 and t2 not in uniq: uniq.append(t2)
    return uniq

def _to_close_frame(df_like, name=None):
    if df_like is None or len(df_like) == 0:
        return pd.DataFrame()
    if "Close" in df_like:
        close = df_like["Close"]
    else:
        close = df_like
    if isinstance(close, pd.Series):
        close = close.to_frame(name or close.name or "Close")
    if hasattr(close, "columns") and isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]
    return close

def _stooq_symbol_for(t: str) -> str:
    # Stooq usa sufijo .US para la mayoría de acciones/ETFs de USA
    if t.startswith("^"):
        # reemplazo de índice: ^GSPC ~ SPY.US como proxy
        return "SPY.US"
    # si ya trae un sufijo (p.ej. .US), respeta
    if "." in t:
        return t
    return f"{t}.US"

@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices_resilient(tickers, start=None, end=None, interval="1d", source_pref="Auto"):
    """
    Devuelve (precios_df, fallidos). Pipeline:
    1) Yahoo batch
    2) Yahoo por ticker
    3) Stooq con mapeo .US (y ^GSPC -> SPY.US)
    """
    tickers = _clean_tickers(tickers)
    if not tickers:
        return pd.DataFrame(), []

    # batch (rápido) con yfinance.Tickers().history()
    def try_batch_yf(tks):
        try:
            yh = yf.Tickers(" ".join(tks))
            if start is None and end is None:
                h = yh.history(period="max", interval=interval, auto_adjust=True)
            else:
                h = yh.history(start=start, end=end, interval=interval, auto_adjust=True)
            close = _to_close_frame(h)
            if not close.empty:
                keep = [c for c in close.columns if c in tks]
                return close[keep]
        except Exception:
            pass
        return pd.DataFrame()

    # per-ticker (con más fallback)
    def try_one_yf(t):
        try:
            if start is None and end is None:
                d = yf.download(t, period="max", interval=interval, auto_adjust=True, progress=False)
            else:
                d = yf.download(t, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            d = _to_close_frame(d, name=t).dropna(how="all")
            if not d.empty:
                if t not in d.columns and d.shape[1] == 1:
                    d.columns = [t]
                return d[[t]]
            # último recurso con objeto Ticker
            h = yf.Ticker(t).history(period="max", auto_adjust=True)
            if not h.empty and "Close" in h:
                h = h[["Close"]].rename(columns={"Close": t})
                if start or end:
                    s = pd.to_datetime(start) if start else pd.Timestamp.min
                    e = pd.to_datetime(end) if end else pd.Timestamp.max
                    h = h.loc[(h.index >= s) & (h.index <= e)]
                return h
        except Exception:
            pass
        return pd.DataFrame()

    def try_one_stooq(t):
        try:
            import pandas_datareader.data as pdr
            sym = _stooq_symbol_for(t)
            s = pd.to_datetime(start) if start else (datetime.utcnow()-timedelta(days=365*5))
            e = pd.to_datetime(end) if end else datetime.utcnow()
            df = pdr.DataReader(sym, "stooq", start=s, end=e)
            if df is not None and not df.empty and "Close" in df:
                return df.sort_index()[["Close"]].rename(columns={"Close": t})
        except Exception:
            pass
        return pd.DataFrame()

    # 1) Yahoo batch
    prices = try_batch_yf(tickers)
    got = set(prices.columns) if not prices.empty else set()
    missing = [t for t in tickers if t not in got]

    # 2) Yahoo per ticker
    per_ticker_frames = []
    for t in missing:
        d = try_one_yf(t)
        if not d.empty:
            per_ticker_frames.append(d)

    # 3) Stooq .US para los que sigan faltando
    still_missing = [t for t in missing if t not in (set([c for df in per_ticker_frames for c in df.columns]))]
    for t in still_missing:
        d = try_one_stooq(t)
        if not d.empty:
            per_ticker_frames.append(d)

    if not prices.empty or per_ticker_frames:
        frames = []
        if not prices.empty: frames.append(prices)
        if per_ticker_frames: frames += per_ticker_frames
        out = pd.concat(frames, axis=1).sort_index()
        out = out.loc[:, out.notna().any()]
        failed = [t for t in tickers if t not in out.columns]
        return out, failed

    return pd.DataFrame(), tickers

@st.cache_data(show_spinner=False, ttl=600)
def last_prices_resilient(tickers, source_pref="Auto"):
    tickers = _clean_tickers(tickers)
    ok, failed = {}, []
    for t in tickers:
        px = np.nan
        # YF
        try:
            d = yf.download(t, period="7d", interval="1d", auto_adjust=True, progress=False)
            c = _to_close_frame(d, name=t).ffill()
            if not c.empty:
                px = float(c.iloc[-1,0])
        except Exception:
            pass
        # Stooq fallback (.US)
        if (np.isnan(px)) and (source_pref in ("Auto","Stooq")):
            try:
                import pandas_datareader.data as pdr
                sym = _stooq_symbol_for(t)
                s = datetime.utcnow()-timedelta(days=14)
                e = datetime.utcnow()
                df = pdr.DataReader(sym, "stooq", start=s, end=e)
                if df is not None and not df.empty and "Close" in df:
                    px = float(df.sort_index()["Close"].iloc[-1])
            except Exception:
                pass
        if np.isnan(px): failed.append(t)
        else: ok[t] = px
    return ok, failed

# ============== MÉTRICAS =========================
def annualize_return(d, freq=252):
    if d.empty: return np.nan
    return float((1+d).prod()**(freq/max(len(d),1)) - 1)
def annualize_vol(d, freq=252):
    if d.empty: return np.nan
    return float(d.std(ddof=0)*np.sqrt(freq))
def sharpe(d, rf=0.0, freq=252):
    if d.empty: return np.nan
    er=annualize_return(d,freq); ev=annualize_vol(d,freq)
    return (er-rf)/ev if ev and ev>0 else np.nan
def sortino(d, rf=0.0, freq=252):
    if d.empty: return np.nan
    neg=d.copy(); neg[neg>0]=0
    dd=np.sqrt((neg**2).mean())*np.sqrt(freq)
    er=annualize_return(d,freq)
    return (er-rf)/dd if dd and dd>0 else np.nan
def max_drawdown(cum):
    if cum.empty: return np.nan
    return float((cum/cum.cummax()-1).min())
def calmar(d,freq=252):
    if d.empty: return np.nan
    er=annualize_return(d,freq); mdd=abs(max_drawdown((1+d).cumprod()))
    return er/mdd if mdd and mdd>0 else np.nan

# ============== TX → POSITIONS ====================
def tidy_transactions(tx:pd.DataFrame)->pd.DataFrame:
    if tx.empty: return tx
    df=tx.copy()
    for c in ["TradeID","Account","Ticker","Name","AssetType","Currency",
              "TradeDate","Side","Shares","Price","Fees","Taxes","FXRate",
              "GrossAmount","NetAmount","LotID","Source","Notes"]:
        if c not in df.columns: df[c]=np.nan

    df["Ticker"]=df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","",regex=False)
    df=df.dropna(subset=["Ticker"])
    df["TradeDate"]=pd.to_datetime(df["TradeDate"],errors="coerce").dt.date
    for n in ["Shares","Price","Fees","Taxes","FXRate","GrossAmount","NetAmount"]:
        df[n]=pd.to_numeric(df[n],errors="coerce")
    df["FXRate"]=df["FXRate"].fillna(1.0)

    def signed(row):
        s=str(row.get("Side","")).lower().strip()
        q=float(row.get("Shares",0) or 0)
        if s in ("sell","venta","vender","-1"): return -abs(q)
        return abs(q)
    df["SignedShares"]=df.apply(signed,axis=1)
    return df

def positions_from_tx(tx:pd.DataFrame):
    if tx.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    df=tidy_transactions(tx)
    uniq=sorted(df["Ticker"].unique().tolist())
    if not uniq:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])

    last_map, failed_lp = last_prices_resilient(uniq, source_pref=st.session_state.get("data_source","Auto"))
    pos=[]
    for t,grp in df.groupby("Ticker"):
        sh=float(grp["SignedShares"].sum())
        if abs(sh)<1e-12: continue
        buys=grp["SignedShares"]>0
        if buys.any():
            tot_sh=float(grp.loc[buys,"SignedShares"].sum())
            cost_leg = (grp.loc[buys,"SignedShares"]*grp.loc[buys,"Price"].fillna(0)).sum()
            fees_leg = grp.loc[buys,"Fees"].fillna(0).sum() + grp.loc[buys,"Taxes"].fillna(0).sum()
            tot_cost = cost_leg + fees_leg
            avg=tot_cost/tot_sh if tot_sh>0 else np.nan
        else:
            avg=np.nan
        px=last_map.get(t,np.nan)
        mv= sh*px if not np.isnan(px) else np.nan
        inv= sh*avg if not np.isnan(avg) else np.nan
        pl= mv-inv if not (np.isnan(mv) or np.isnan(inv)) else np.nan
        pos.append([t,sh,avg,inv,px,mv,pl])
    dfp=pd.DataFrame(pos,columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    if failed_lp: st.caption("⚠️ Sin precio reciente para: " + ", ".join(failed_lp))
    return dfp.sort_values("MarketValue",ascending=False)

def weights_from_positions(pos_df:pd.DataFrame):
    if pos_df.empty: return pd.Series(dtype=float)
    total=pos_df["MarketValue"].fillna(0).sum()
    if total<=0: return pd.Series(dtype=float)
    return (pos_df.set_index("Ticker")["MarketValue"]/total).sort_values(ascending=False)

def const_weight_returns(price_df:pd.DataFrame, weights:pd.Series):
    if price_df is None or price_df.empty or weights is None or len(weights)==0:
        return pd.Series(dtype=float), pd.DataFrame()
    cols=[c for c in price_df.columns if c in weights.index]
    if not cols: return pd.Series(dtype=float), pd.DataFrame()
    price_df=price_df[cols].dropna(how="all")
    rets=price_df.pct_change().dropna(how="all")
    if rets.empty: return pd.Series(dtype=float), pd.DataFrame()
    w=weights.reindex(rets.columns).fillna(0).values
    port=(rets*w).sum(axis=1)
    return port, rets

# ============== OPTIMIZACIÓN ====================
def max_sharpe(mean_ret, cov, rf=0.0, bounds=None):
    n=len(mean_ret)
    if n==0: return np.array([])
    if bounds is None: bounds=[(0.0,1.0)]*n
    def neg_sharpe(w):
        mu=float(np.dot(w,mean_ret))
        sig=float(np.sqrt(np.dot(w,np.dot(cov,w))))
        if sig==0: return 9999.0
        return -((mu-rf)/sig)
    cons=[{"type":"eq","fun":lambda w: np.sum(w)-1.0}]
    x0=np.array([1.0/n]*n)
    res=minimize(neg_sharpe,x0,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":500})
    return res.x if (hasattr(res,"success") and res.success) else x0

# ============== LOAD BASE ====================
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","^GSPC",str)
w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)

# ============== SIDEBAR ======================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Herramientas"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
data_source = st.sidebar.selectbox("Fuente de precios", ["Auto","Yahoo!","Stooq"], index=0)
st.session_state["data_source"] = "Auto" if data_source=="Auto" else ("Yahoo" if data_source=="Yahoo!" else "Stooq")

period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ============== HOME / BROKER-LIKE TABLE =====
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")

    if st.button("🔄 Refrescar datos"):
        refresh_data(); st.rerun()

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún. Carga operaciones en **Transactions**.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()

    # Primer intento: Auto (Yahoo→Stooq)
    prices, failed = fetch_prices_resilient(tickers + [benchmark], start=start_date, source_pref=st.session_state["data_source"])

    bench_ret = pd.Series(dtype=float)
    # Si benchmark vino vacío, usa SPY/SPY.US como proxy
    if not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")
    else:
        spy_try, _ = fetch_prices_resilient(["SPY"], start=start_date, source_pref="Yahoo")
        if spy_try.empty:
            spy_try, _ = fetch_prices_resilient(["SPY"], start=start_date, source_pref="Stooq")
        if not spy_try.empty and "SPY" in spy_try.columns:
            bench_ret = spy_try["SPY"].pct_change().dropna()

    # Si nada funcionó, reintenta forzando Stooq para todos
    if (prices.empty or len(prices.columns)==0):
        prices2, failed2 = fetch_prices_resilient(tickers, start=start_date, source_pref="Stooq")
        if not prices2.empty:
            prices, failed = prices2, failed2

    failed_set=set(failed)
    if prices.empty or len(prices.columns)==0:
        msg="No se pudieron obtener precios históricos."
        if failed_set: msg += " Fallidos: " + ", ".join(sorted(failed_set))
        st.warning(msg); st.stop()
    if failed_set:
        st.caption("⚠️ Tickers sin histórico: " + ", ".join(sorted(failed_set)))

    # Retornos por activo (ventana)
    asset_rets = prices.pct_change().dropna(how="all")
    window_change = (prices.ffill().iloc[-1]/prices.ffill().iloc[0]-1).reindex(tickers).fillna(np.nan)

    # P&L desde compra
    since_buy = (pos_df.set_index("Ticker")["MarketPrice"]/pos_df.set_index("Ticker")["AvgCost"] - 1).replace([np.inf,-np.inf], np.nan)

    # Pesos
    w = weights_from_positions(pos_df)
    pos_df = pos_df.set_index("Ticker").loc[w.index].reset_index()

    # Tabla estilo broker
    view = pd.DataFrame({
        "#": np.arange(1, len(w)+1),
        "Ticker": pos_df["Ticker"].values,
        "Shares": pos_df["Shares"].values,
        "Avg Buy": pos_df["AvgCost"].values,
        "Last": pos_df["MarketPrice"].values,
        "P/L $": pos_df["UnrealizedPL"].values,
        "P/L % (compra)": (since_buy.reindex(pos_df["Ticker"]).values*100.0),
        "Δ % ventana": (window_change.reindex(pos_df["Ticker"]).values*100.0),
        "Peso %": (w.reindex(pos_df["Ticker"]).values*100.0),
        "Valor": pos_df["MarketValue"].values
    })

    # Formato y colores
    fmt_money = {"Avg Buy":"$,.2f","Last":"$,.2f","P/L $":"$,.2f","Valor":"$,.2f"}
    fmt_pct = {"P/L % (compra)":"{:.2f}%","Δ % ventana":"{:.2f}%","Peso %":"{:.2f}%"}
    def color_pct(val):
        if pd.isna(val): return "color: inherit;"
        return "color: #18b26b;" if val>=0 else "color: #ff4d4f;"

    styled = (view.style
        .format({**fmt_money, **fmt_pct})
        .applymap(color_pct, subset=["P/L % (compra)","Δ % ventana"])
        .set_properties(subset=["#","Ticker","Shares"], **{"font-weight":"600"})
    )

    c_top1, c_top2 = st.columns([2,1])
    with c_top1:
        st.subheader("Composición y rendimiento")
        port_ret, _ = const_weight_returns(prices, w)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Rend. anualizado", f"{(annualize_return(port_ret) or 0)*100:,.2f}%")
        c2.metric("Vol. anualizada", f"{(annualize_vol(port_ret) or 0)*100:,.2f}%")
        c3.metric("Sharpe", f"{(sharpe(port_ret, rf) or 0):.2f}")
        cum=(1+port_ret).cumprod(); mdd=max_drawdown(cum)
        c4.metric("Max Drawdown", f"{(mdd or 0)*100:,.2f}%")
        c5.metric("Sortino", f"{(sortino(port_ret, rf) or 0):.2f}")
        c6.metric("Calmar", f"{(calmar(port_ret) or 0):.2f}")

        curve=pd.DataFrame({"Portafolio":(1+port_ret).cumprod()})
        if not bench_ret.empty:
            curve["Benchmark"]=(1+bench_ret).cumprod().reindex(curve.index).ffill()
        st.plotly_chart(px.line(curve, title="Crecimiento de 1.0"), use_container_width=True)

    with c_top2:
        alloc = pd.DataFrame({"Ticker":w.index,"Weight":w.values})
        st.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", title="Asignación"), use_container_width=True)

    st.subheader("📋 Detalle de posiciones (estilo broker)")
    st.dataframe(styled, use_container_width=True, height=min(600, 120 + 32*len(view)))

    # Descargar CSV
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar posiciones (CSV)", csv, file_name="mi_portafolio.csv", mime="text/csv")

# ============== OPTIMIZAR (tolerante a fallas y reintento Stooq) ===========
elif page=="Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones."); st.stop()

    tickers = pos_df["Ticker"].tolist()

    # 1) Yahoo
    prices, failed = fetch_prices_resilient(tickers, start=start_date, source_pref="Yahoo")

    # 2) Si no hay nada aún, reintenta con Stooq (.US)
    if prices.empty:
        prices, failed = fetch_prices_resilient(tickers, start=start_date, source_pref="Stooq")

    if prices.empty:
        st.warning("No hay precios para optimizar. " + (", ".join(failed) if failed else ""))
        st.stop()

    if failed:
        st.caption("⚠️ Excluidos por falta de histórico: " + ", ".join(sorted(failed)))
        good = [c for c in prices.columns if prices[c].notna().any()]
        prices = prices[good]
        pos_df = pos_df[pos_df["Ticker"].isin(good)].copy()
        tickers = good

    w_cur = weights_from_positions(pos_df)
    port_ret_cur, asset_rets = const_weight_returns(prices, w_cur)
    if asset_rets.empty:
        st.warning("No hay retornos suficientes para optimizar.")
        st.stop()

    mean_daily=asset_rets.mean(); cov=asset_rets.cov(); mu_ann=(1+mean_daily)**252-1
    w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
    w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)
    bounds=[(w_min,w_max)]*len(tickers)
    w_opt = max_sharpe(mu_ann.values, cov.values, rf=rf, bounds=bounds)
    w_opt = pd.Series(w_opt, index=tickers).clip(lower=w_min, upper=w_max); w_opt/=w_opt.sum()

    port_ret_opt,_ = const_weight_returns(prices, w_opt)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sharpe actual", f"{(sharpe(port_ret_cur, rf) or 0):.2f}")
    c2.metric("Sharpe propuesto", f"{(sharpe(port_ret_opt, rf) or 0):.2f}")
    c3.metric("Vol. actual", f"{(annualize_vol(port_ret_cur) or 0)*100:,.2f}%")
    c4.metric("Vol. propuesto", f"{(annualize_vol(port_ret_opt) or 0)*100:,.2f}%")

    compare=pd.DataFrame({"Weight Actual":w_cur,"Weight Propuesto":w_opt}).fillna(0)
    compare["Δ (pp)"]=(compare["Weight Propuesto"]-compare["Weight Actual"])*100
    st.dataframe(compare.style.format({"Weight Actual":"{:.2%}","Weight Propuesto":"{:.2%}","Δ (pp)":"{:.2f}"}), use_container_width=True)

# ============== EVALUAR CANDIDATO ==================
elif page=="Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("Sin posiciones."); st.stop()

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ","")
    if not tkr: st.stop()
    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_resilient(sorted(set(tickers+[tkr])), start=start_date, source_pref=st.session_state["data_source"])
    if prices.empty:
        st.warning("No se pudieron descargar precios. "+(", ".join(failed) if failed else "")); st.stop()

    w_cur = weights_from_positions(pos_df)
    last_map,_ = last_prices_resilient([tkr], source_pref=st.session_state["data_source"])
    last=last_map.get(tkr,np.nan)
    if np.isnan(last): st.warning("Ticker inválido o sin precio reciente."); st.stop()

    mode = st.radio("Forma de evaluación", ["Asignar porcentaje","Añadir acciones"], horizontal=True)
    if mode=="Asignar porcentaje":
        pct=st.slider("Peso objetivo del candidato",0.0,0.40,0.10,0.01)
        w_new=(w_cur*(1-pct)).reindex(prices.columns).fillna(0.0); w_new[tkr]+=pct
    else:
        qty=st.number_input("Acciones a comprar (simulado)",min_value=1,value=5,step=1)
        pv=pos_df["MarketValue"].sum(); add=qty*last; base=pv+add if (pv+add)>0 else 1.0
        w_new=(w_cur*pv)/base; w_new=w_new.reindex(prices.columns).fillna(0.0); w_new[tkr]+=add/base

    port_cur,_=const_weight_returns(prices, w_cur.reindex(prices.columns, fill_value=0))
    port_new,_=const_weight_returns(prices, w_new.reindex(prices.columns, fill_value=0))
    c=st.columns(4)
    sh_old=sharpe(port_cur,rf); sh_new=sharpe(port_new,rf)
    c[0].metric("Sharpe actual", f"{(sh_old or 0):.2f}")
    c[1].metric("Sharpe con candidato", f"{(sh_new or 0):.2f}",
                delta=None if (pd.isna(sh_old) or pd.isna(sh_new)) else f"{(sh_new-sh_old):.2f}")
    cum_cur=(1+port_cur).cumprod(); cum_new=(1+port_new).cumprod()
    mdd_cur=max_drawdown(cum_cur); mdd_new=max_drawdown(cum_new)
    c[2].metric("MDD actual", f"{(0 if pd.isna(mdd_cur) else mdd_cur)*100:,.2f}%")
    c[3].metric("MDD con candidato", f"{(0 if pd.isna(mdd_new) else mdd_new)*100:,.2f}%",
                delta=None if (pd.isna(mdd_cur) or pd.isna(mdd_new)) else f"{((mdd_new-mdd_cur)*100):.2f}%")

# ============== EXPLORAR =========================
elif page=="Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr=st.text_input("Ticker", value="MSFT").upper().strip().replace(" ","")
    if tkr:
        hist=yf.download(tkr,period="1y",interval="1d",auto_adjust=True,progress=False)
        if not hist.empty:
            fig=go.Figure(data=[go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"])])
            fig.update_layout(title=f"Velas: {tkr}", xaxis_rangeslider_visible=False, height=480)
            st.plotly_chart(fig, use_container_width=True)

# ============== HERRAMIENTAS =====================
elif page=="Herramientas":
    st.title("🧰 Herramientas")
    st.write("Próximamente: lotes detallados, ventas parciales y P/L realizado vs no realizado.")
