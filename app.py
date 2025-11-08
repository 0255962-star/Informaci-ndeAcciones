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

def _ws(name:str):
    return gc.open_by_key(SHEET_ID).worksheet(name)

# ============== SCHEMA-ROBUST LOAD ==============
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

# ============== PRICE LAYER (Resiliente por ticker) ==============
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
        if t2 and t2 not in uniq: uniq.append(t2)
    return uniq

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
            return df.sort_index()[["Close"]].rename(columns={"Close": t})
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices_resilient(tickers, start=None, end=None, interval="1d", source_pref="Auto"):
    uniq = _clean_tickers(tickers)
    if not uniq: return pd.DataFrame(), []
    if start is None: start=(datetime.utcnow()-timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end is None: end=datetime.utcnow().strftime("%Y-%m-%d")
    series, failed = {}, []

    def try_all(t):
        if source_pref in ("Auto","Yahoo"):
            d=_fetch_one_yf(t,start,end,interval)
            if not d.empty: return d
        if source_pref in ("Auto","Stooq"):
            d=_fetch_one_stooq(t,start,end)
            if not d.empty: return d
        return pd.DataFrame()

    for t in uniq:
        d=try_all(t)
        if d.empty: failed.append(t)
        else:
            if t not in d.columns and d.shape[1]==1: d.columns=[t]
            series[t]=d

    if not series: return pd.DataFrame(), failed
    df = pd.concat([series[k] for k in sorted(series.keys())], axis=1).sort_index()
    df = df.loc[:, df.notna().any()]
    return df, failed

def last_prices_resilient(tickers, source_pref="Auto"):
    uniq=_clean_tickers(tickers)
    if not uniq: return {}, []
    ok, failed = {}, []

    def try_yf(t):
        try:
            d = yf.download(t, period="5d", interval="1d", auto_adjust=True, progress=False)
            if not d.empty:
                c = _flatten_close(d, [t]).ffill().dropna(how="all")
                if not c.empty: return float(c.iloc[-1,0])
            hist = yf.Ticker(t).history(period="10d", auto_adjust=True)
            if not hist.empty and "Close" in hist:
                return float(hist["Close"].dropna().iloc[-1])
        except Exception: pass
        return np.nan

    def try_stooq(t):
        try:
            import pandas_datareader.data as pdr
            d = pdr.DataReader(t, "stooq", start=datetime.utcnow()-timedelta(days=10), end=datetime.utcnow())
            if d is not None and not d.empty and "Close" in d:
                return float(d.sort_index()["Close"].dropna().iloc[-1])
        except Exception: pass
        return np.nan

    for t in uniq:
        px=np.nan
        if source_pref in ("Auto","Yahoo"): px=try_yf(t)
        if np.isnan(px) and source_pref in ("Auto","Stooq"): px=try_stooq(t)
        if np.isnan(px): failed.append(t)
        else: ok[t]=px
    return ok, failed

# ============== METRICS ==============
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
def tracking_error(p,b,freq=252):
    r=(p-b).dropna()
    return float(r.std(ddof=0)*np.sqrt(freq)) if not r.empty else np.nan

# ============== TX → POSITIONS ==============
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
            # costo ponderado real (comisiones/impuestos incluidos si vinieran en NetAmount)
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

# ============== OPTIMIZER ==============
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

# ============== GEMINI (opcional) ==============
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

# ============== LOAD BASE ==============
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","^GSPC",str)
w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)

# ============== SIDEBAR ==============
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Herramientas"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
data_source = st.sidebar.selectbox("Fuente de precios", ["Auto","Yahoo!","Stooq"], index=0)
st.session_state["data_source"] = "Auto" if data_source=="Auto" else ("Yahoo" if data_source=="Yahoo!" else "Stooq")

period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ============== HOME / BROKER-LIKE TABLE ==============
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")

    if st.button("🔄 Refrescar datos"):
        refresh_data(); st.rerun()

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty:
        st.info("No hay posiciones aún. Carga operaciones en **Transactions**.")
        st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_resilient(tickers + [benchmark], start=start_date, source_pref=st.session_state["data_source"])

    bench_ret = pd.Series(dtype=float)
    if not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")

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

    # Sugerencias simples
    suggestions=[]
    if (alloc["Weight"]**2).sum()>0.15: suggestions.append("Concentración alta (HHI>0.15). Considera diversificar.")
    if suggestions:
        for s in suggestions: st.info(s)

# ============== OPTIMIZAR ==============
elif page=="Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("No hay posiciones."); st.stop()
    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_resilient(tickers, start=start_date, source_pref=st.session_state["data_source"])
    if prices.empty: st.warning("No hay precios para optimizar. " + (", ".join(failed) if failed else "")); st.stop()

    w_cur = weights_from_positions(pos_df)
    port_ret_cur, asset_rets = const_weight_returns(prices, w_cur)
    if asset_rets.empty: st.warning("No hay retornos suficientes."); st.stop()

    mean_daily=asset_rets.mean(); cov=asset_rets.cov(); mu_ann=(1+mean_daily)**252-1
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

# ============== EVALUAR =================
elif page=="Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("Sin posiciones."); st.stop()

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ","")
    if not tkr: st.stop()
    tickers = pos_df["Ticker"].tolist()
    prices, failed = fetch_prices_resilient(sorted(set(tickers+[tkr])), start=start_date, source_pref=st.session_state["data_source"])
    if prices.empty: st.warning("No se pudieron descargar precios. "+(", ".join(failed) if failed else "")); st.stop()

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
    c[1].metric("Sharpe con candidato", f"{(sh_new or 0):.2f}", delta=None if (pd.isna(sh_old) or pd.isna(sh_new)) else f"{(sh_new-sh_old):.2f}")
    cum_cur=(1+port_cur).cumprod(); cum_new=(1+port_new).cumprod()
    mdd_cur=max_drawdown(cum_cur); mdd_new=max_drawdown(cum_new)
    c[2].metric("MDD actual", f"{(0 if pd.isna(mdd_cur) else mdd_cur)*100:,.2f}%")
    c[3].metric("MDD con candidato", f"{(0 if pd.isna(mdd_new) else mdd_new)*100:,.2f}%",
                delta=None if (pd.isna(mdd_cur) or pd.isna(mdd_new)) else f"{((mdd_new-mdd_cur)*100):.2f}%")

# ============== RESEARCH =================
elif page=="Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr=st.text_input("Ticker", value="MSFT").upper().strip().replace(" ","")
    if tkr:
        hist=yf.download(tkr,period="1y",interval="1d",auto_adjust=True,progress=False)
        if not hist.empty:
            fig=go.Figure(data=[go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"])])
            fig.update_layout(title=f"Velas: {tkr}", xaxis_rangeslider_visible=False, height=480)
            st.plotly_chart(fig, use_container_width=True)

# ============== TOOLS ====================
elif page=="Herramientas":
    st.title("🧰 Herramientas")
    st.write("Próximamente: lotes detallados, ventas parciales y P/L realizado vs no realizado.")
