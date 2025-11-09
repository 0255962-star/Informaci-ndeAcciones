# APP Finanzas – Portafolio Activo (SOLO Yahoo Finance)
# Parches clave:
# - User-Agent real (YF_USER_AGENT) para evitar bloqueos en Streamlit Cloud
# - yf.download(..., raise_errors=False, repair=True, timeout=25, threads=False)
# - Fallback por-ticker + Ticker().history() con backoff
# - Si ^GSPC falla, seguimos sin romper (se informa en UI)

from datetime import datetime, timedelta
import os, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

import gspread
from google.oauth2.service_account import Credentials

# ============== CONFIG YF: USER-AGENT / TIMEOUTS ==============
# Forzamos un User-Agent aceptable para evitar bloqueos de Yahoo
os.environ.setdefault(
    "YF_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

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
section[data-testid="stSidebar"] { border-right: 1px solid #1e2435; }
div[data-testid="stMetricValue"]{font-size:1.6rem}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""", unsafe_allow_html=True)

# ============== SECRETS / SHEETS ==============
SHEET_ID = st.secrets.get("SHEET_ID") or st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
GCP_SA = st.secrets.get("gcp_service_account", {})
if not SHEET_ID:
    st.error("Falta `SHEET_ID` en secrets."); st.stop()
if not GCP_SA:
    st.error("Falta el bloque `gcp_service_account` en secrets."); st.stop()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
try:
    credentials = Credentials.from_service_account_info(GCP_SA, scopes=SCOPES)
    gc = gspread.authorize(credentials)
except Exception as e:
    st.error(f"No se pudo autorizar Google Sheets: {e}"); st.stop()

# ============== CACHES ==============
def refresh_all():
    for f in (read_sheet, load_all_data, fetch_prices_yf, last_prices_yf):
        try: f.clear()
        except Exception: pass
    try: st.cache_data.clear()
    except Exception: pass

@st.cache_data(ttl=600, show_spinner=False)
def read_sheet(name:str)->pd.DataFrame:
    sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet(name)
    return pd.DataFrame(ws.get_all_records())

@st.cache_data(ttl=600, show_spinner=False)
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

# ============== HELPERS YF ==============
def _clean_tickers(tickers):
    uniq=[]
    for t in tickers:
        if not isinstance(t,str): continue
        t2=t.upper().strip().replace(" ","")
        if t2 and t2 not in uniq: uniq.append(t2)
    return uniq

def _flatten_close(df_like):
    if df_like is None or len(df_like)==0:
        return pd.DataFrame()
    df = df_like.copy()

    # MultiIndex (Close, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close = df['Close'].copy()
        except Exception:
            # fallback si no hay nivel 'Close'
            return pd.DataFrame()
        close = close.loc[:, close.notna().any()]
        return close

    # Columnas planas con Close
    if 'Close' in df.columns:
        return df[['Close']].rename(columns={'Close':'_TMP_'}).copy()

    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices_yf(tickers, start=None, end=None, interval="1d",
                     max_retries=2, pause_sec=0.9, min_rows=3):
    """
    SOLO Yahoo Finance:
      1) Lote con yf.download(lista)
      2) Per-ticker con yf.download()
      3) Último recurso: Ticker().history()

    Flags críticos: raise_errors=False, repair=True, timeout=25, threads=False
    """
    errs = []
    tickers = _clean_tickers(tickers)
    if not tickers:
        return pd.DataFrame(), [], errs

    # -------- 1) LOTE
    batch = pd.DataFrame()
    try:
        if start is None and end is None:
            batch = yf.download(
                tickers=tickers, period="max", interval=interval,
                auto_adjust=True, progress=False, group_by="ticker",
                threads=False, raise_errors=False, repair=True, timeout=25
            )
        else:
            batch = yf.download(
                tickers=tickers, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, group_by="ticker",
                threads=False, raise_errors=False, repair=True, timeout=25
            )
    except Exception as e:
        errs.append(f"[batch] {type(e).__name__}: {e}")

    frames=[]; got=set()
    if batch is not None and not batch.empty:
        flat = _flatten_close(batch)
        if not flat.empty:
            if list(flat.columns)==['_TMP_'] and len(tickers)==1:
                flat.columns=[tickers[0]]
            frames.append(flat)
            got=set(flat.columns)

    missing=[t for t in tickers if t not in got]

    # -------- 2) POR-TICKER
    def _download_one(t):
        for k in range(max_retries+1):
            try:
                d = yf.download(
                    t,
                    period="max" if (start is None and end is None) else None,
                    start=start, end=end, interval=interval,
                    auto_adjust=True, progress=False, threads=False,
                    raise_errors=False, repair=True, timeout=25
                )
                close = _flatten_close(d)
                if not close.empty:
                    if list(close.columns)==['_TMP_']: close.columns=[t]
                    if close.shape[0] >= min_rows:
                        return close[[t]] if t in close.columns else close
            except Exception as e:
                if k == max_retries:
                    errs.append(f"[download:{t}] {type(e).__name__}: {e}")
            time.sleep(pause_sec*(k+1))
        return pd.DataFrame()

    for t in missing:
        df_t=_download_one(t)
        if not df_t.empty:
            frames.append(df_t)

    if frames:
        prices=pd.concat(frames,axis=1).sort_index()
        prices=prices.loc[:, prices.notna().any()]
        failed=[t for t in tickers if t not in prices.columns]
        return prices, failed, errs

    # -------- 3) HISTORY OBJETO
    frames=[]
    for t in missing:
        succ=False
        for k in range(max_retries+1):
            try:
                th=yf.Ticker(t).history(period="max", auto_adjust=True, repair=True, raise_errors=False, timeout=25)
                if th is not None and not th.empty and "Close" in th.columns:
                    ser=th["Close"].rename(t).to_frame()
                    if start or end:
                        s=pd.to_datetime(start) if start else pd.Timestamp.min
                        e=pd.to_datetime(end) if end else pd.Timestamp.max
                        ser=ser.loc[(ser.index>=s)&(ser.index<=e)]
                    if ser.shape[0] >= min_rows:
                        frames.append(ser); succ=True; break
            except Exception as e:
                if k == max_retries:
                    errs.append(f"[history:{t}] {type(e).__name__}: {e}")
            time.sleep(pause_sec*(k+1))
        if not succ:
            errs.append(f"[history:{t}] sin datos suficientes")

    if frames:
        prices=pd.concat(frames,axis=1).sort_index()
        prices=prices.loc[:, prices.notna().any()]
        failed=[t for t in tickers if t not in prices.columns]
        return prices, failed, errs

    return pd.DataFrame(), tickers, errs

@st.cache_data(ttl=600, show_spinner=False)
def last_prices_yf(tickers, max_retries=2, pause_sec=0.7):
    tickers=_clean_tickers(tickers)
    ok, failed={}, []

    def add_from_df(t, df):
        try:
            close=_flatten_close(df)
            if not close.empty:
                if list(close.columns)==['_TMP_']: close.columns=[t]
                s=close.iloc[:,0].dropna().ffill()
                if not s.empty:
                    ok[t]=float(s.iloc[-1]); return True
        except Exception:
            pass
        return False

    for t in tickers:
        success=False
        for k in range(max_retries+1):
            try:
                d=yf.download(t, period="15d", interval="1d", auto_adjust=True,
                              progress=False, threads=False, raise_errors=False, repair=True, timeout=25)
                if add_from_df(t, d): success=True; break
            except Exception:
                pass
            time.sleep(pause_sec*(k+1))
        if not success:
            for k in range(max_retries+1):
                try:
                    h=yf.Ticker(t).history(period="15d", auto_adjust=True, repair=True, raise_errors=False, timeout=25)
                    if add_from_df(t, h): success=True; break
                except Exception:
                    pass
                time.sleep(pause_sec*(k+1))
        if not success: failed.append(t)

    return ok, failed

# ============== MÉTRICAS =================
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

# ============== TX → POSITIONS =============
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
    last_map, failed_lp = last_prices_yf(uniq)
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
    if failed_lp: st.caption("⚠️ Sin último precio (Yahoo): " + ", ".join(failed_lp))
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

# ============== OPTIMIZACIÓN =============
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

# ============== CARGA BASE ==============
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","^GSPC",str)
w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)

# ============== SIDEBAR ==============
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Diagnóstico"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ============== MI PORTAFOLIO ==============
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")
    if st.button("🔄 Refrescar datos"): refresh_all(); st.rerun()

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("No hay posiciones. Carga operaciones en 'Transactions'."); st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed, errs = fetch_prices_yf(tickers + [benchmark], start=start_date)

    bench_ret = pd.Series(dtype=float)
    if not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")
    elif benchmark in (failed or []):
        st.caption(f"ℹ️ Benchmark '{benchmark}' no disponible ahora en Yahoo. Se omite temporalmente.")

    if prices.empty or len(prices.columns)==0:
        st.warning("No se pudieron obtener precios históricos desde Yahoo.")
        if failed: st.caption("Fallidos: " + ", ".join(sorted(set(failed))))
        if errs:
            with st.expander("Detalles de errores Yahoo"):
                for e in errs: st.code(e)
        st.stop()

    # Retornos, pesos y tabla estilo broker
    asset_rets = prices.pct_change().dropna(how="all")
    window_change = (prices.ffill().iloc[-1]/prices.ffill().iloc[0]-1).reindex(tickers).fillna(np.nan)
    w = weights_from_positions(pos_df)
    since_buy = (pos_df.set_index("Ticker")["MarketPrice"]/pos_df.set_index("Ticker")["AvgCost"] - 1).replace([np.inf,-np.inf], np.nan)
    pos_df = pos_df.set_index("Ticker").loc[w.index].reset_index()

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
        st.subheader("Composición y rendimiento (constante)")
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
    st.download_button("⬇️ Descargar posiciones (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="mi_portafolio.csv", mime="text/csv")

# ============== OPTIMIZAR ==============
elif page=="Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("No hay posiciones."); st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, failed, errs = fetch_prices_yf(tickers, start=start_date)

    if prices.empty:
        st.warning("No hay precios para optimizar desde Yahoo.")
        if failed: st.caption("Fallidos: " + ", ".join(sorted(set(failed))))
        if errs:
            with st.expander("Detalles de errores Yahoo"):
                for e in errs: st.code(e)
        st.stop()

    if failed:
        st.caption("⚠️ Excluidos por falta de histórico: " + ", ".join(sorted(set(failed))))
        good = [c for c in prices.columns if prices[c].notna().any()]
        prices = prices[good]
        pos_df = pos_df[pos_df["Ticker"].isin(good)].copy()
        tickers = good

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
    c4.metric("Vol. propuesto", f"{(annualize_vol(port_ret_opt) or 0)*100:,.2f}%")

    compare=pd.DataFrame({"Weight Actual":w_cur,"Weight Propuesto":w_opt}).fillna(0)
    compare["Δ (pp)"]=(compare["Weight Propuesto"]-compare["Weight Actual"])*100
    st.dataframe(compare.style.format({"Weight Actual":"{:.2%}","Weight Propuesto":"{:.2%}","Δ (pp)":"{:.2f}"}), use_container_width=True)

# ============== EVALUAR CANDIDATO ==============
elif page=="Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("Sin posiciones."); st.stop()

    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ","")
    if not tkr: st.stop()
    tickers = pos_df["Ticker"].tolist()
    prices, failed, errs = fetch_prices_yf(sorted(set(tickers+[tkr])), start=start_date)
    if prices.empty:
        st.warning("No se pudieron descargar precios desde Yahoo.")
        if failed: st.caption("Fallidos: " + ", ".join(sorted(set(failed))))
        if errs:
            with st.expander("Detalles de errores Yahoo"):
                for e in errs: st.code(e)
        st.stop()

    w_cur = weights_from_positions(pos_df)
    last_map,_ = last_prices_yf([tkr])
    last=last_map.get(tkr,np.nan)
    if np.isnan(last): st.warning("Ticker inválido o sin último precio (Yahoo)."); st.stop()

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
                delta=None if (pd.isna(sh_old) or pd.isna(sh_new)) else f"{(sh_new-sh_old):/ .2f}")
    cum_cur=(1+port_cur).cumprod(); cum_new=(1+port_new).cumprod()
    mdd_cur=max_drawdown(cum_cur); mdd_new=max_drawdown(cum_new)
    c[2].metric("MDD actual", f"{(0 if pd.isna(mdd_cur) else mdd_cur)*100:,.2f}%")
    c[3].metric("MDD con candidato", f"{(0 if pd.isna(mdd_new) else mdd_new)*100:,.2f}%",
                delta=None if (pd.isna(mdd_cur) or pd.isna(mdd_new)) else f"{((mdd_new-mdd_cur)*100):.2f}%")

# ============== EXPLORAR / DIAGNÓSTICO ==============
elif page=="Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr=st.text_input("Ticker", value="MSFT").upper().strip().replace(" ","")
    if tkr:
        hist, failed, errs = fetch_prices_yf([tkr], start=start_date)
        if not hist.empty and tkr in hist.columns:
            ser = hist[tkr].dropna()
            if not ser.empty:
                df = pd.DataFrame({"Close": ser})
                fig=go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df["Close"], high=df["Close"], low=df["Close"], close=df["Close"]
                )])
                fig.update_layout(title=f"Serie (Close) {tkr}", xaxis_rangeslider_visible=False, height=480)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Yahoo no entregó histórico para ese ticker.")
            if failed: st.caption("Fallidos: " + ", ".join(failed))
            if errs:
                with st.expander("Detalles de errores Yahoo"):
                    for e in errs: st.code(e)

elif page=="Diagnóstico":
    st.title("🩺 Diagnóstico rápido")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Secrets")
        st.write({"SHEET_ID": bool(SHEET_ID), "gcp_service_account": bool(GCP_SA)})
        try:
            sh = gc.open_by_key(SHEET_ID)
            titles = [w.title for w in sh.worksheets()]
            st.success("Hojas encontradas:"); st.write(titles)
        except Exception as e:
            st.error(f"No pude listar hojas: {e}")

        st.subheader("Settings")
        st.dataframe(load_all_data()[1], use_container_width=True)

    with col2:
        st.subheader("Ping Yahoo (SPY 30 días)")
        try:
            test = yf.download("SPY", period="30d", interval="1d", auto_adjust=True,
                               progress=False, threads=False, raise_errors=False, repair=True, timeout=25)
            if test is None or test.empty:
                st.warning("Sin datos de Yahoo para SPY (posible rate-limit/bloqueo de red).")
            else:
                st.success(f"OK: {len(test)} barras."); st.dataframe(test.tail(5))
        except Exception as e:
            st.error(f"Error al consultar Yahoo: {e}")

    st.subheader("Transactions (muestra)")
    st.dataframe(load_all_data()[0].head(10), use_container_width=True)
