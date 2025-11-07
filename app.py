import os
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

# ================== CONFIG ==================
st.set_page_config(page_title="APP Finanzas – Portafolio Activo", page_icon="💼", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetricValue"]{font-size:1.6rem}
.block-container{padding-top:1rem}
</style>
""", unsafe_allow_html=True)

# ================== SECRETS ==================
SHEET_ID = st.secrets.get("SHEET_ID") or st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
GCP_SA = st.secrets.get("gcp_service_account", {})
SHOW_DEBUG = bool(st.secrets.get("SHOW_DEBUG", False))

if not SHEET_ID or not GCP_SA:
    st.error("Faltan secretos: `SHEET_ID` y/o `gcp_service_account`.")
    st.stop()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(GCP_SA, scopes=SCOPES)
gc = gspread.authorize(credentials)

# ================== SHEETS HELPERS ==================
@st.cache_data(show_spinner=False, ttl=600)
def read_sheet(name:str)->pd.DataFrame:
    sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet(name)
    df = pd.DataFrame(ws.get_all_records())
    return df

def write_sheet_append(name:str, row:list):
    sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet(name)
    ws.append_row(row, value_input_option="USER_ENTERED")

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
    tx = safe("Transactions", ["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"])
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

# ================== DATA FETCH (Robusto) ==================
def _flatten_close(df_or_panel, expected):
    if df_or_panel is None or len(df_or_panel)==0:
        return pd.DataFrame()
    close = df_or_panel["Close"] if "Close" in df_or_panel else df_or_panel
    if isinstance(close, pd.Series):
        nm = expected[0] if expected else (close.name if close.name else "Close")
        close = close.to_frame(name=nm)
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]
    if expected:
        close = close.loc[:, [c for c in close.columns if c in expected]]
    return close

def _clean_tickers(tickers):
    uniq = []
    for t in tickers:
        if not isinstance(t,str): continue
        t2 = t.upper().strip().replace(" ", "")
        if t2 and t2 not in uniq:
            uniq.append(t2)
    return uniq

@st.cache_data(show_spinner=False, ttl=900)
def _fetch_bulk_yf(tickers, start, end, interval):
    try:
        df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False, threads=True)
        return _flatten_close(df, tickers).dropna(how="all")
    except Exception:
        return pd.DataFrame()

def _fetch_one_yf(t, start, end, interval):
    try:
        d = yf.download(t, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        d = _flatten_close(d, [t]).dropna(how="all")
        if not d.empty: return d.rename(columns={d.columns[0]: t})
        hist = yf.Ticker(t).history(period="max", auto_adjust=True)
        if not hist.empty and "Close" in hist:
            hist = hist[["Close"]].rename(columns={"Close": t})
            return hist.loc[(hist.index>=pd.to_datetime(start)) & (hist.index<=pd.to_datetime(end))]
    except Exception:
        pass
    return pd.DataFrame()

def _fetch_one_stooq(t, start, end):
    try:
        import pandas_datareader.data as pdr
        df = pdr.DataReader(t, "stooq", start=pd.to_datetime(start), end=pd.to_datetime(end))
        if df is None or df.empty: return pd.DataFrame()
        if "Close" in df:
            df = df.sort_index()[["Close"]].rename(columns={"Close": t})
            return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_prices_resilient(tickers, start=None, end=None, interval="1d", source_pref="Auto"):
    """
    source_pref: "Auto" (Yahoo→Stooq), "Yahoo", "Stooq"
    Devuelve (prices_df, failed_list)
    """
    uniq = _clean_tickers(tickers)
    if not uniq: return pd.DataFrame(), []

    if start is None:
        start = (datetime.utcnow() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    failed = []
    pieces = []

    def try_yahoo():
        bulk = _fetch_bulk_yf(uniq, start, end, interval)
        have = set(bulk.columns) if not bulk.empty else set()
        pcs = []
        if not bulk.empty: pcs.append(bulk)
        miss = []
        for t in uniq:
            if t in have: continue
            d = _fetch_one_yf(t, start, end, interval)
            if d.empty: miss.append(t)
            else: pcs.append(d)
        return pcs, miss

    def try_stooq(miss_list=None):
        base = uniq if miss_list is None else miss_list
        pcs = []; miss = []
        for t in base:
            d = _fetch_one_stooq(t, start, end)
            if d.empty: miss.append(t)
            else: pcs.append(d)
        return pcs, miss

    if source_pref == "Yahoo":
        pcs, miss = try_yahoo()
        if pcs:
            df = pd.concat(pcs, axis=1).sort_index()
            df = df.loc[:, df.notna().any()]
            return (df if not df.empty else pd.DataFrame()), miss if df.empty else [t for t in miss if t not in df.columns]
        return pd.DataFrame(), uniq

    if source_pref == "Stooq":
        pcs, miss = try_stooq()
        if pcs:
            df = pd.concat(pcs, axis=1).sort_index()
            df = df.loc[:, df.notna().any()]
            return (df if not df.empty else pd.DataFrame()), miss if df.empty else [t for t in miss if t not in df.columns]
        return pd.DataFrame(), uniq

    # Auto: Yahoo → Stooq para faltantes
    pcs, miss = try_yahoo()
    if miss:
        pcs2, miss2 = try_stooq(miss)
        pcs += pcs2
        failed = miss2
    if pcs:
        df = pd.concat(pcs, axis=1).sort_index()
        df = df.loc[:, df.notna().any()]
        if df.empty:
            return pd.DataFrame(), uniq
        return df, failed
    return pd.DataFrame(), uniq

def last_prices_resilient(tickers, source_pref="Auto"):
    uniq = _clean_tickers(tickers)
    if not uniq: return {}, []
    ok, failed = {}, []

    def last_from_df(df, cols_expected):
        nonlocal ok
        if df is not None and not df.empty:
            close = _flatten_close(df, cols_expected).ffill().dropna(how="all")
            if close is not None and not close.empty:
                row = close.tail(1)
                for c in row.columns:
                    v = row.iloc[0][c]
                    if pd.notna(v): ok[c] = float(v)

    tried_stooq = False

    if source_pref in ("Auto","Yahoo"):
        try:
            df = yf.download(uniq, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
            last_from_df(df, uniq)
        except Exception:
            pass
        for t in uniq:
            if t in ok: continue
            d = _fetch_one_yf(t, (datetime.utcnow()-timedelta(days=10)).strftime("%Y-%m-%d"), datetime.utcnow().strftime("%Y-%m-%d"), "1d")
            if not d.empty:
                ok[t] = float(d.iloc[-1, 0])

    if source_pref in ("Auto","Stooq"):
        import pandas_datareader.data as pdr  # necesita estar en requirements
        tried_stooq = True
        for t in uniq:
            if t in ok: continue
            try:
                d = pdr.DataReader(t, "stooq", start=datetime.utcnow()-timedelta(days=10), end=datetime.utcnow())
                if d is not None and not d.empty and "Close" in d:
                    ok[t] = float(d.sort_index()["Close"].dropna().iloc[-1])
            except Exception:
                pass

    for t in uniq:
        if t not in ok: failed.append(t)
    if tried_stooq and failed:
        # Algunos símbolos (por ejemplo ^GSPC) no existen en Stooq; se reportan como fallidos.
        pass
    return ok, failed

# ================== METRICS ==================
def annualize_return(d, freq=252):
    if d.empty: return np.nan
    return float((1+d).prod()**(freq/max(len(d),1)) - 1)

def annualize_vol(d, freq=252):
    if d.empty: return np.nan
    return float(d.std(ddof=0)*np.sqrt(freq))

def sharpe(d, rf=0.0, freq=252):
    if d.empty: return np.nan
    er = annualize_return(d,freq); ev = annualize_vol(d,freq)
    return (er-rf)/ev if ev and ev>0 else np.nan

def sortino(d, rf=0.0, freq=252):
    if d.empty: return np.nan
    neg = d.copy(); neg[neg>0]=0
    dd = np.sqrt((neg**2).mean())*np.sqrt(freq)
    er = annualize_return(d,freq)
    return (er-rf)/dd if dd and dd>0 else np.nan

def max_drawdown(cum):
    if cum.empty: return np.nan
    return float((cum/cum.cummax()-1).min())

def calmar(d, freq=252):
    if d.empty: return np.nan
    er = annualize_return(d,freq)
    mdd = abs(max_drawdown((1+d).cumprod()))
    return er/mdd if mdd and mdd>0 else np.nan

def tracking_error(p,b,freq=252):
    r=(p-b).dropna()
    return float(r.std(ddof=0)*np.sqrt(freq)) if not r.empty else np.nan

def information_ratio(p,b,rf=0.0,freq=252):
    te = tracking_error(p,b,freq)
    if not te or np.isnan(te): return np.nan
    return (annualize_return(p,freq)-annualize_return(b,freq))/te

def regression_beta_alpha(p,b,freq=252):
    df = pd.concat([p,b],axis=1).dropna()
    if df.empty: return np.nan, np.nan
    X=df.iloc[:,1].values; Y=df.iloc[:,0].values
    X_=np.vstack([np.ones_like(X),X]).T
    a,beta = np.linalg.lstsq(X_,Y,rcond=None)[0]
    alpha_ann = (1 + a)**freq - 1
    return float(beta), float(alpha_ann)

# ================== PORTFOLIO ==================
def tidy_transactions(tx:pd.DataFrame)->pd.DataFrame:
    if tx.empty: return tx
    df = tx.copy()
    for c in ["Ticker","TradeDate","Side","Shares","Price","Fees","Notes"]:
        if c not in df.columns: df[c]=np.nan
    df["Ticker"]=(df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","",regex=False))
    df["Ticker"].replace({"":np.nan}, inplace=True)
    df = df.dropna(subset=["Ticker"])
    df["TradeDate"]=pd.to_datetime(df["TradeDate"],errors="coerce").dt.date
    def signed(r):
        side=str(r.get("Side","")).strip().lower()
        q=float(r.get("Shares",0) or 0)
        if side in ("sell","venta","vender","-1"): return -abs(q)
        if side in ("buy","compra","1"): return  abs(q)
        return q
    df["SignedShares"]=df.apply(signed,axis=1)
    return df

def positions_from_tx(tx:pd.DataFrame):
    if tx.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    df = tidy_transactions(tx)
    uniq = sorted(df["Ticker"].unique().tolist())
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
            tot_cost=float((grp.loc[buys,"SignedShares"]*grp.loc[buys,"Price"].fillna(0)).sum()+grp.loc[buys,"Fees"].fillna(0).sum())
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

# ================== OPTIMIZACIÓN ==================
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

# ================== GEMINI (opcional) ==================
def gemini_translate(text, target_lang="es"):
    if not GEMINI_API_KEY or not text: return text
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model=genai.GenerativeModel("gemini-1.5-flash")
        resp=model.generate_content(f"Traduce al {target_lang} y resume en 80-120 palabras, tono claro:\n\n{text}")
        return resp.text.strip()
    except Exception:
        return text

# ================== CARGA BASE ==================
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","^GSPC",str)
w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)
fee_per_trade = get_setting(settings_df,"FeePerTrade",0.0,float)
slippage_bps = get_setting(settings_df,"SlippageBps",0,float)

# ================== UI: Sidebar ==================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Herramientas"])
window = st.sidebar.selectbox("Ventana histórica", ["1Y","3Y","5Y","Max"], index=1)
data_source = st.sidebar.selectbox("Fuente de precios", ["Auto","Yahoo!","Stooq"], index=0)
st.session_state["data_source"] = "Auto" if data_source=="Auto" else ("Yahoo" if data_source=="Yahoo!" else "Stooq")

period_map={"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ================== HOME ==================
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún. Registra operaciones en **Transactions**.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()

    prices, failed = fetch_prices_resilient(tickers+[benchmark], start=start_date, source_pref=st.session_state["data_source"])
    bench_series = pd.Series(dtype=float)
    if not prices.empty and benchmark in prices.columns:
        bench_series = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")
    elif data_source!="Stooq" and "SPY" not in tickers:
        # si falta benchmark, intenta proxy SPY
        alt, _ = fetch_prices_resilient(tickers+["SPY"], start=start_date, source_pref=st.session_state["data_source"])
        if not alt.empty and "SPY" in alt.columns:
            bench_series = alt["SPY"].pct_change().dropna()
            prices = alt.drop(columns=["SPY"], errors="ignore")

    if prices.empty:
        msg = "No se pudieron obtener precios históricos."
        if failed: msg += " Tickers con problemas: " + ", ".join(failed)
        st.warning(msg + " Revisa símbolos / fuente seleccionada o intenta más tarde.")
        st.stop()

    w = weights_from_positions(pos_df)
    if w.empty:
        st.warning("No se pudieron calcular pesos de mercado."); st.stop()

    port_ret, asset_rets = const_weight_returns(prices, w)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Rend. anualizado", f"{(annualize_return(port_ret) or 0)*100:,.2f}%")
    c2.metric("Vol. anualizada", f"{(annualize_vol(port_ret) or 0)*100:,.2f}%")
    c3.metric("Sharpe", f"{(sharpe(port_ret, rf) or 0):.2f}")
    c4.metric("Sortino", f"{(sortino(port_ret, rf) or 0):.2f}")
    cum=(1+port_ret).cumprod(); mdd=max_drawdown(cum)
    c5.metric("Max Drawdown", f"{(mdd or 0)*100:,.2f}%")
    c6.metric("Calmar", f"{(calmar(port_ret) or 0):.2f}")

    if not bench_series.empty:
        te=tracking_error(port_ret, bench_series); ir=information_ratio(port_ret, bench_series, rf)
        st.caption(f"Tracking Error: **{(te or 0)*100:,.2f}%** · Information Ratio: **{(ir or 0):.2f}**")

    perf=pd.DataFrame({"Portafolio":(1+port_ret).cumprod()})
    if not bench_series.empty:
        perf["Benchmark"]=(1+bench_series).cumprod().reindex(perf.index).ffill()
    st.plotly_chart(px.line(perf, title="Crecimiento de 1.0"), use_container_width=True)

    alloc=w.reset_index(); alloc.columns=["Ticker","Weight"]
    st.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", title="Asignación actual"), use_container_width=True)

    cols=["Ticker","Shares","AvgCost","MarketPrice","MarketValue","UnrealizedPL"]
    st.dataframe(pos_df[cols].style.format({"AvgCost":"$,.2f","MarketPrice":"$,.2f","MarketValue":"$,.2f","UnrealizedPL":"$,.2f"}), use_container_width=True)

    st.subheader("🧭 Sugerencias")
    suggestions=[]
    hhi=(alloc["Weight"]**2).sum()
    if hhi>0.15: suggestions.append("Concentración alta (HHI > 0.15). Considera diversificar.")
    cal=calmar(port_ret)
    if not np.isnan(cal) and cal<0.3: suggestions.append("Calmar bajo; revisa drawdowns/volatilidad.")
    if suggestions:
        for s in suggestions: st.info(s)
    else:
        st.success("Sin alertas para el periodo seleccionado.")

# ================== OPTIMIZAR ==================
elif page=="Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún."); st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_resilient(tickers, start=start_date, source_pref=st.session_state["data_source"])
    if prices.empty:
        st.warning("No hay precios para optimizar. " + ("Fallas: "+", ".join(failed) if failed else "")); st.stop()

    w_cur = weights_from_positions(pos_df)
    port_ret_cur, asset_rets = const_weight_returns(prices, w_cur)
    if asset_rets.empty:
        st.warning("No hay retornos suficientes para optimizar."); st.stop()

    mean_daily=asset_rets.mean(); cov=asset_rets.cov(); mu_ann=(1+mean_daily)**252-1
    bounds=[(w_min,w_max)]*len(tickers)
    w_opt = max_sharpe(mu_ann.values, cov.values, rf=rf, bounds=bounds)
    w_opt = pd.Series(w_opt, index=tickers).clip(lower=w_min, upper=w_max); w_opt/=w_opt.sum()

    port_ret_opt,_ = const_weight_returns(prices, w_opt)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sharpe actual", f"{(sharpe(port_ret_cur, rf) or 0):.2f}")
    c2.metric("Sharpe propuesto", f"{(sharpe(port_ret_opt, rf) or 0):.2f}")
    c3.metric("Vol. actual", f"{(annualize_vol(port_ret_cur) or 0)*100:,.2f}%")
    c4.metric("Vol. propuesta", f"{(annualize_vol(port_ret_opt) or 0)*100:,.2f}%")

    compare=pd.DataFrame({"Weight Actual":w_cur,"Weight Propuesto":w_opt}).fillna(0)
    compare["Δ (pp)"]=(compare["Weight Propuesto"]-compare["Weight Actual"])*100
    st.dataframe(compare.style.format({"Weight Actual":"{:.2%}","Weight Propuesto":"{:.2%}","Δ (pp)":"{:.2f}"}), use_container_width=True)

    st.subheader("Órdenes sugeridas (simulación)")
    pv=pos_df["MarketValue"].sum()
    orders=[]
    for t in tickers:
        px=float(pos_df.loc[pos_df["Ticker"]==t,"MarketPrice"].fillna(0).values[0])
        if px<=0: continue
        tgt_val=pv*w_opt.get(t,0.0); tgt_sh=np.floor(tgt_val/px)
        cur_sh=float(pos_df.loc[pos_df["Ticker"]==t,"Shares"].values[0])
        delta=tgt_sh-cur_sh
        if delta!=0:
            side="Buy" if delta>0 else "Sell"
            est=abs(delta)*px + fee_per_trade + (abs(delta)*px)*(slippage_bps/10000)
            orders.append([t,side,int(delta),px,est])
    ord_df=pd.DataFrame(orders,columns=["Ticker","Side","Shares","EstPrice","EstCost"])
    if ord_df.empty:
        st.success("Tu portafolio ya está muy cercano al objetivo propuesto.")
    else:
        st.dataframe(ord_df.style.format({"EstPrice":"$,.2f","EstCost":"$,.2f"}), use_container_width=True)
        with st.expander("Confirmar (escribe en Transactions)"):
            if st.button("Registrar todas"):
                today=datetime.utcnow().date().strftime("%Y-%m-%d")
                for _,r in ord_df.iterrows():
                    write_sheet_append("Transactions",[r["Ticker"],today,r["Side"],r["Shares"],r["EstPrice"],0.0,"Auto-rebalance simulado"])
                st.success("Órdenes registradas. Pulsa R para refrescar.")

# ================== EVALUAR CANDIDATO ==================
elif page == "Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("Sin posiciones.")
        st.stop()

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ", "")
    if not tkr:
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    all_tickers = sorted(set(tickers + [tkr]))

    prices, failed = fetch_prices_resilient(
        all_tickers, start=start_date, source_pref=st.session_state.get("data_source", "Auto")
    )
    if prices.empty:
        st.warning("No se pudieron descargar precios para el candidato. " + ("Fallas: " + ", ".join(failed) if failed else ""))
        st.stop()

    w_cur = weights_from_positions(pos_df)
    alloc_mode = st.radio("Forma de evaluación", ["Asignar porcentaje", "Añadir acciones"], horizontal=True)

    last_map, _ = last_prices_resilient([tkr], source_pref=st.session_state.get("data_source", "Auto"))
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

    # Retornos
    port_cur, _ = const_weight_returns(prices, w_cur.reindex(prices.columns, fill_value=0))
    port_new, assets_new = const_weight_returns(prices, w_new.reindex(prices.columns, fill_value=0))

    # Métricas (protegidas contra NaN)
    sh_old = sharpe(port_cur, rf)
    sh_new = sharpe(port_new, rf)
    cum_cur = (1 + port_cur).cumprod() if not port_cur.empty else pd.Series(dtype=float)
    cum_new = (1 + port_new).cumprod() if not port_new.empty else pd.Series(dtype=float)
    mdd_cur = max_drawdown(cum_cur) if not cum_cur.empty else np.nan
    mdd_new = max_drawdown(cum_new) if not cum_new.empty else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe actual", f"{(sh_old or 0):.2f}")
    c2.metric("Sharpe con candidato", f"{(sh_new or 0):.2f}",
              delta=None if (pd.isna(sh_old) or pd.isna(sh_new)) else f"{(sh_new - sh_old):.2f}")
    c3.metric("MDD actual", f"{(0 if pd.isna(mdd_cur) else mdd_cur)*100:,.2f}%")
    c4.metric("MDD con candidato", f"{(0 if pd.isna(mdd_new) else mdd_new)*100:,.2f}%",
              delta=None if (pd.isna(mdd_cur) or pd.isna(mdd_new)) else f"{((mdd_new - mdd_cur)*100):.2f}%")

    # Correlación candidato vs cartera
    rho = np.nan
    if tkr in assets_new.columns and not port_cur.empty:
        rho = assets_new[tkr].corr(port_cur.reindex(assets_new.index))
        if not np.isnan(rho):
            st.caption(f"Correlación candidato vs. cartera: **{rho:.2f}**")

    # Regla de decisión (umbral simple)
    delta_sharpe = (sh_new if not pd.isna(sh_new) else np.nan) - (sh_old if not pd.isna(sh_old) else np.nan)
    pass_rule = (
        (not pd.isna(delta_sharpe) and delta_sharpe >= 0.03)
        and (not pd.isna(mdd_cur) and not pd.isna(mdd_new) and (mdd_new - mdd_cur) >= -0.02)
        and (pd.isna(rho) or rho <= 0.75)
    )

    # 👇 Importante: if/else explícito (sin operador ternario) para evitar el bug de Streamlit
    if pass_rule:
        st.success("✅ Recomendación positiva.")
    else:
        st.warning("⚠️ La mejora no supera los umbrales definidos.")

    with st.expander("Registrar compra simulada en Transactions"):
        if st.button("Registrar (Buy)"):
            today = datetime.utcnow().date().strftime("%Y-%m-%d")
            qty_to_log = 5 if alloc_mode == "Añadir acciones" else 0
            write_sheet_append("Transactions", [tkr, today, "Buy", qty_to_log, last, 0.0, "Evaluación aprobada"])
            st.success("Operación registrada.")
            
# ================== EXPLORAR ==================
elif page=="Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr=st.text_input("Ticker", value="MSFT").upper().strip().replace(" ","")
    if tkr:
        hist=yf.download(tkr,period="1y",interval="1d",auto_adjust=True,progress=False)
        if hist.empty:
            st.warning("No hay datos para ese ticker."); st.stop()
        fig=go.Figure(data=[go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"])])
        fig.update_layout(title=f"Velas: {tkr}", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        df=hist.copy(); df["SMA20"]=df["Close"].rolling(20).mean(); df["SMA50"]=df["Close"].rolling(50).mean()
        st.plotly_chart(px.line(df[["Close","SMA20","SMA50"]], title="SMAs"), use_container_width=True)
        info={}; 
        try: info=yf.Ticker(tkr).info or {}
        except Exception: info={}
        long_name=info.get("longName") or tkr; summary=info.get("longBusinessSummary") or ""
        st.subheader(long_name); st.write(gemini_translate(summary,"es") if summary else "Sin descripción disponible.")

# ================== HERRAMIENTAS ==================
elif page=="Herramientas":
    st.title("🧰 Herramientas")
    tabs=st.tabs(["β / α","CAPM rápido","Monte Carlo"])

    with tabs[0]:
        st.subheader("β / α vs Benchmark")
        pos_df=positions_from_tx(tx_df)
        if pos_df.empty: st.info("Sin posiciones.")
        else:
            tickers=pos_df["Ticker"].tolist()
            prices, _ = fetch_prices_resilient(tickers+[benchmark], start=start_date, source_pref=st.session_state["data_source"])
            if prices.empty or benchmark not in prices.columns:
                prices, _ = fetch_prices_resilient(tickers+["SPY"], start=start_date, source_pref=st.session_state["data_source"])
                if prices.empty or "SPY" not in prices.columns:
                    st.warning("No hay datos suficientes del benchmark.")
                else:
                    core=[c for c in prices.columns if c!="SPY"]
                    w=weights_from_positions(pos_df)
                    port,_=const_weight_returns(prices[core], w)
                    bench=prices["SPY"].pct_change().dropna()
                    if port.empty or bench.empty: st.warning("Datos insuficientes.")
                    else:
                        beta,alpha=regression_beta_alpha(port, bench)
                        c1,c2=st.columns(2); c1.metric("Beta", f"{(beta or 0):.2f}"); c2.metric("Alpha anualizado", f"{(alpha or 0)*100:,.2f}%")
            else:
                core=[c for c in prices.columns if c!=benchmark]
                w=weights_from_positions(pos_df)
                port,_=const_weight_returns(prices[core], w)
                bench=prices[[benchmark]].pct_change().dropna()[benchmark]
                if port.empty or bench.empty: st.warning("Datos insuficientes.")
                else:
                    beta,alpha=regression_beta_alpha(port, bench)
                    c1,c2=st.columns(2); c1.metric("Beta", f"{(beta or 0):.2f}"); c2.metric("Alpha anualizado", f"{(alpha or 0)*100:,.2f}%")

    with tabs[1]:
        st.subheader("CAPM rápido")
        exp_mkt=st.number_input("E[Rm] esperado (anual)", value=0.08, step=0.01, format="%.2f")
        beta_i=st.number_input("Beta del activo", value=1.00, step=0.10, format="%.2f")
        rf_capm=st.number_input("r_f (anual)", value=float(rf), step=0.005, format="%.3f")
        er=rf_capm+beta_i*(exp_mkt-rf_capm)
        st.metric("E[Ri] (CAPM)", f"{er*100:,.2f}%")

    with tabs[2]:
        st.subheader("Simulación Monte Carlo (portafolio)")
        pos_df=positions_from_tx(tx_df)
        if pos_df.empty: st.info("Sin posiciones.")
        else:
            tickers=pos_df["Ticker"].tolist()
            prices,_=fetch_prices_resilient(tickers, start=start_date, source_pref=st.session_state["data_source"])
            if prices.empty: st.warning("No hay precios para simular.")
            else:
                w=weights_from_positions(pos_df)
                port,assets=const_weight_returns(prices, w)
                if port.empty: st.warning("No hay retornos suficientes.")
                else:
                    mu=port.mean(); sigma=port.std(ddof=0)
                    horizon=st.slider("Horizonte (días)",30,365,180,10)
                    sims=st.slider("N simulaciones",200,5000,1000,100)
                    rng=np.random.default_rng(42)
                    sim_end=np.array([np.prod(1+rng.normal(mu, sigma, size=horizon)) for _ in range(sims)])
                    p5,p50,p95=np.percentile(sim_end,[5,50,95])
                    c1,c2,c3=st.columns(3)
                    c1.metric("P5", f"{(p5-1)*100:,.2f}%"); c2.metric("P50", f"{(p50-1)*100:,.2f}%"); c3.metric("P95", f"{(p95-1)*100:,.2f}%")
                    st.plotly_chart(px.histogram(sim_end-1, nbins=40, title="Distribución de rendimientos simulados"), use_container_width=True)
