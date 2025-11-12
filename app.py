# APP Finanzas – Portafolio con cache incremental + backfill en Google Sheets
from datetime import datetime, timedelta, timezone
import os, time, logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import yfinance as yf
from scipy.optimize import minimize
import gspread
from google.oauth2.service_account import Credentials

# ================== CONFIG GENERAL ==================
st.set_page_config(page_title="APP Finanzas – Portafolio Activo", page_icon="💼", layout="wide")
st.markdown("""
<style>
:root {
  --bg: #0b0f1a; --card: #141927; --muted: #8b93a7;
  --brand: #7c6cff; --up: #18b26b; --down: #ff4d4f;
}
.block-container { padding-top: 0.8rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #1e2435; }
div[data-testid="stMetricValue"]{font-size:1.6rem}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* última col (botón eliminar) chica y encabezado rojo */
th:has(> div:contains("➖")) { color: #ff4d4f !important; width: 48px !important; }
td:has(input[type="checkbox"]) { text-align: center; }
</style>
""", unsafe_allow_html=True)

# Silenciar logger ruidoso de yfinance
try:
    yf_logger = logging.getLogger("yfinance")
    yf_logger.setLevel(logging.CRITICAL)
    yf.utils.get_yf_logger().setLevel(logging.CRITICAL)
except Exception:
    pass

# ================== SECRETS / SHEETS ==================
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

# ================== CACHES DE STREAMLIT ==================
def refresh_all():
    for f in (
        read_sheet, load_all_data, fetch_prices_yahoo_direct,
        fetch_prices_yf_batch, last_prices_direct, last_prices_yf_safe,
        cache_read_prices, cache_latest_date_per_ticker, cache_earliest_date_per_ticker
    ):
        try: f.clear()
        except Exception: pass
    try: st.cache_data.clear()
    except Exception: pass

@st.cache_data(ttl=600, show_spinner=False)
def read_sheet(name:str)->pd.DataFrame:
    sh = gc.open_by_key(SHEET_ID); ws = sh.worksheet(name)
    return pd.DataFrame(ws.get_all_records())

def get_worksheet(name:str):
    sh = gc.open_by_key(SHEET_ID); return sh.worksheet(name)

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

def _clean_tickers(tickers):
    uniq=[]
    for t in tickers:
        if not isinstance(t,str): continue
        t2=t.upper().strip().replace(" ","")
        if t2 and t2 not in uniq: uniq.append(t2)
    return uniq

# ================== CACHE EN GOOGLE SHEETS (PricesCache) ==================
CACHE_SHEET_NAME = "PricesCache"   # columnas A:C -> Date | Ticker | Close

def _ensure_cache_sheet():
    """Crea la hoja PricesCache si no existe, con encabezados."""
    sh = gc.open_by_key(SHEET_ID)
    titles = [w.title for w in sh.worksheets()]
    if CACHE_SHEET_NAME not in titles:
        ws = sh.add_worksheet(title=CACHE_SHEET_NAME, rows=1000, cols=3)
        ws.update("A1:C1", [["Date","Ticker","Close"]])

def _get_cache_ws():
    _ensure_cache_sheet()
    sh = gc.open_by_key(SHEET_ID)
    return sh.worksheet(CACHE_SHEET_NAME)

@st.cache_data(ttl=0, show_spinner=False)
def cache_read_prices(tickers, start_date=None):
    """
    Lee de PricesCache y devuelve un DataFrame wide (index=fecha, cols=tickers).
    start_date: 'YYYY-MM-DD' o None.
    """
    ws = _get_cache_ws()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(values[1:], columns=values[0])
    df = df[(df["Date"]!="") & (df["Ticker"]!="") & (df["Close"]!="")]
    if df.empty: 
        return pd.DataFrame()
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Ticker","Close"])

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date).normalize()]

    if tickers:
        tset = {t.upper().strip().replace(" ","") for t in tickers}
        df = df[df["Ticker"].isin(tset)]

    if df.empty:
        return pd.DataFrame()

    wide = (
        df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
          .sort_index()
    )
    wide = wide[~wide.index.duplicated(keep="last")]
    return wide

@st.cache_data(ttl=0, show_spinner=False)
def cache_latest_date_per_ticker(tickers):
    """Regresa dict {ticker: ultima_fecha_en_cache} (o None si no hay)."""
    ws = _get_cache_ws()
    values = ws.get_all_values()
    out = {t: None for t in tickers}
    if not values or len(values) < 2:
        return out
    df = pd.DataFrame(values[1:], columns=values[0])
    if df.empty: 
        return out
    df = df[(df["Date"]!="") & (df["Ticker"]!="")]
    if df.empty: 
        return out
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    df = df.dropna(subset=["Date","Ticker"])
    for t in tickers:
        t2 = t.upper().strip().replace(" ","")
        sub = df.loc[df["Ticker"]==t2, "Date"]
        out[t] = sub.max() if not sub.empty else None
    return out

@st.cache_data(ttl=0, show_spinner=False)
def cache_earliest_date_per_ticker(tickers):
    """Regresa dict {ticker: primera_fecha_en_cache} (o None si no hay)."""
    ws = _get_cache_ws()
    values = ws.get_all_values()
    out = {t: None for t in tickers}
    if not values or len(values) < 2:
        return out
    df = pd.DataFrame(values[1:], columns=values[0])
    if df.empty: 
        return out
    df = df[(df["Date"]!="") & (df["Ticker"]!="")]
    if df.empty: 
        return out
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    df = df.dropna(subset=["Date","Ticker"])
    for t in tickers:
        t2 = t.upper().strip().replace(" ","")
        sub = df.loc[df["Ticker"]==t2, "Date"]
        out[t] = sub.min() if not sub.empty else None
    return out

def cache_append_prices(df_wide):
    """
    Recibe un DataFrame wide (index fechas, columnas tickers, valores Close)
    y lo inserta (append) en PricesCache, evitando duplicados básicos.
    """
    if df_wide is None or df_wide.empty:
        return 0
    ws = _get_cache_ws()

    # normaliza index
    df_wide = df_wide.copy()
    df_wide.index = pd.to_datetime(df_wide.index).normalize()

    # Leemos cache existente para deduplicar
    existing = cache_read_prices(list(df_wide.columns))
    to_write = df_wide.copy()

    if existing is not None and not existing.empty:
        merged = pd.concat([existing, to_write]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        diff = merged.loc[to_write.index, to_write.columns].where(~existing.reindex(merged.index).notna(), other=np.nan)
        to_write = diff.dropna(how="all")

    if to_write.empty:
        return 0

    rows = []
    for dt, row in to_write.sort_index().iterrows():
        for t, val in row.items():
            if pd.isna(val): 
                continue
            rows.append([dt.strftime("%Y-%m-%d"), str(t), float(val)])

    BATCH = 1000
    total = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i+BATCH]
        ws.append_rows(chunk, value_input_option="RAW")
        total += len(chunk)
    return total

def fetch_from_yf_missing(tickers, start_map, end_date=None):
    """
    Descarga de yfinance SOLO lo faltante hacia adelante por ticker.
    start_map: dict {ticker: fecha_inicio} (si None -> 10 años).
    end_date: str 'YYYY-MM-DD' o None.
    Devuelve DF wide de Close.
    """
    starts = [d for d in start_map.values() if d is not None]
    if starts:
        min_start = min(starts)
    else:
        min_start = (datetime.utcnow() - timedelta(days=365*10)).strftime("%Y-%m-%d")

    tickers_clean = [t.upper().strip().replace(" ","") for t in tickers]
    try:
        d = yf.download(" ".join(tickers_clean),
                        start=min_start, end=end_date, interval="1d",
                        auto_adjust=True, progress=False, group_by="ticker", threads=False)
    except Exception:
        return pd.DataFrame()

    if d is None or d.empty:
        return pd.DataFrame()

    frames=[]
    for t in tickers_clean:
        try:
            sub = d[(t,"Close")] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
            s = pd.Series(sub).dropna().rename(t)
            if not s.empty:
                s_start = start_map.get(t)
                if s_start is not None:
                    start_dt = pd.to_datetime(s_start).normalize() + pd.Timedelta(days=1)
                    s = s[s.index.normalize() >= start_dt]
                frames.append(s.to_frame())
        except Exception:
            continue
    if frames:
        out = pd.concat(frames, axis=1).sort_index()
        out.index = pd.to_datetime(out.index).normalize()
        return out
    return pd.DataFrame()

def fetch_backfill_yf(tickers, start_date, end_date):
    """
    Descarga histórico HACIA ATRÁS cuando start_date < primera fecha en cache.
    """
    if start_date is None or end_date is None:
        return pd.DataFrame()
    tickers_clean = [t.upper().strip().replace(" ","") for t in tickers]
    try:
        d = yf.download(" ".join(tickers_clean),
                        start=start_date, end=end_date, interval="1d",
                        auto_adjust=True, progress=False, group_by="ticker", threads=False)
    except Exception:
        return pd.DataFrame()
    if d is None or d.empty:
        return pd.DataFrame()
    frames=[]
    for t in tickers_clean:
        try:
            sub = d[(t,"Close")] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
            s = pd.Series(sub).dropna().rename(t)
            if not s.empty:
                frames.append(s.to_frame())
        except Exception:
            continue
    if frames:
        out = pd.concat(frames, axis=1).sort_index()
        out.index = pd.to_datetime(out.index).normalize()
        return out
    return pd.DataFrame()

# ================== YAHOO DIRECTO (fallback) ==================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
os.environ.setdefault("YF_USER_AGENT", UA)

def _unix(dt): return int(pd.Timestamp(dt, tz=timezone.utc).timestamp())

def _parse_chart_json(js) -> pd.Series:
    result = js.get("chart", {}).get("result", [])
    if not result: return pd.Series(dtype=float)
    r = result[0]
    ts = r.get("timestamp", [])
    if not ts: return pd.Series(dtype=float)
    idx = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert(None).normalize()
    try:
        adj = r.get("indicators", {}).get("adjclose", [])
        if adj and "adjclose" in adj[0]:
            vals = adj[0]["adjclose"]
            return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception:
        pass
    try:
        quote = r.get("indicators", {}).get("quote", [])
        if quote and "close" in quote[0]:
            vals = quote[0]["close"]
            return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

def _direct_one(ticker, start=None, end=None, interval="1d", timeout=25):
    params = {"interval": interval, "includeAdjustedClose": "true",
              "events": "div,splits", "lang": "en-US", "region": "US"}
    if start is None and end is None:
        params["range"] = "max"
    else:
        if start: params["period1"] = _unix(start)
        params["period2"] = int(pd.Timestamp.utcnow().timestamp())
    headers = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*"}
    last_err = None
    for host in HOSTS:
        url = f"{host}/v8/finance/chart/{ticker}"
        for k in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
                if r.status_code != 200:
                    last_err = f"{url} -> HTTP {r.status_code}"; time.sleep(0.7*(k+1)); continue
                js = r.json()
                s = _parse_chart_json(js)
                if isinstance(s, pd.Series) and not s.empty and len(s.dropna()) >= 3:
                    s.name = ticker; return s, None
                last_err = f"{url} -> respuesta vacía"
            except Exception as e:
                last_err = f"{url} -> {type(e).__name__}: {e}"
            time.sleep(0.7*(k+1))
    return pd.Series(dtype=float), (last_err or "sin datos")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices_yahoo_direct(tickers, start=None, end=None, interval="1d"):
    tickers = _clean_tickers(tickers)
    if not tickers: return pd.DataFrame(), [], []
    frames=[]; failed=[]; errs=[]
    for t in tickers:
        s, err = _direct_one(t, start=start, end=end, interval=interval)
        if isinstance(s, pd.Series) and not s.empty:
            frames.append(s.rename(t).to_frame())
        else:
            failed.append(t); 
            if err: errs.append(f"[direct:{t}] {err}")
    if frames:
        df = pd.concat(frames, axis=1).sort_index()
        df.index = pd.to_datetime(df.index).normalize()
        return df, failed, errs
    return pd.DataFrame(), tickers, errs

# ================== yfinance BATCH (solo últimos precios) ==================
@st.cache_data(ttl=1800, show_spinner=False)
def last_prices_yf_safe(tickers):
    tickers = _clean_tickers(tickers)
    if not tickers: return {}, []
    out, failed = {}, []
    try:
        d = yf.download(" ".join(tickers), period="15d", interval="1d",
                        auto_adjust=True, progress=False, group_by="ticker", threads=False)
        if d is None or d.empty: raise RuntimeError("batch vacío")
        for t in tickers:
            try:
                sub = d[(t,"Close")] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
                s = pd.Series(sub).dropna()
                if not s.empty: out[t] = float(s.iloc[-1])
                else: failed.append(t)
            except Exception:
                failed.append(t)
    except Exception:
        failed = tickers
    return out, failed

@st.cache_data(ttl=600, show_spinner=False)
def last_prices_direct(tickers):
    tickers = _clean_tickers(tickers)
    out, failed = {}, []
    for t in tickers:
        s, err = _direct_one(t, interval="1d")
        if isinstance(s, pd.Series) and not s.empty:
            out[t] = float(s.dropna().iloc[-1])
        else:
            failed.append(t)
    return out, failed

# ================== MÉTRICAS ==================
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

# ================== TX → POSITIONS ==================
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

    last_map, failed_lp = last_prices_yf_safe(uniq)
    if failed_lp:
        last_map2, failed_lp2 = last_prices_direct(failed_lp)
        last_map.update(last_map2); failed_lp = failed_lp2

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
    if failed_lp: st.caption("⚠️ Sin último precio: " + ", ".join(failed_lp))
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

# ================== OPTIMIZACIÓN (SLSQP) ==================
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

# ================== ELIMINAR TRANSACCIONES POR TICKER ==================
def delete_transactions_by_ticker(ticker:str)->int:
    ws = get_worksheet("Transactions")
    values = ws.get_all_values()
    if not values: return 0
    headers = values[0]
    try:
        tcol = headers.index("Ticker")
    except ValueError:
        st.error("No encontré la columna 'Ticker' en Transactions."); return 0
    to_delete = []
    for i, row in enumerate(values[1:], start=2):
        if len(row) > tcol and str(row[tcol]).strip().upper() == ticker.strip().upper():
            to_delete.append(i)
    for ridx in reversed(to_delete):
        ws.delete_rows(ridx)
    return len(to_delete)

# ================== CARGA BASE Y CONFIG ==================
tx_df, settings_df, watch_df = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","SPY",str)
w_min = get_setting(settings_df,"MinWeightPerAsset",0.0,float)
w_max = get_setting(settings_df,"MaxWeightPerAsset",0.30,float)

# ================== SIDEBAR ==================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Diagnóstico"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ================== CARGA DE PRECIOS (cache + backfill + forward-fill) ==================
def load_prices_with_fallback(tickers, bench, start_date):
    """
    1) Lee cache (PricesCache) para tickers + benchmark.
    2) Si ventana pedida (start_date) < primera fecha en cache -> BACKFILL.
    3) Si falta desde última fecha hasta hoy -> FORWARD-INCREMENTAL.
    4) Devuelve (prices_sin_benchmark, benchmark_returns, failed, errs).
    """
    all_tickers = list(dict.fromkeys((tickers or []) + [bench]))

    # 1) leer cache existente
    cached = cache_read_prices(all_tickers, start_date=start_date)

    # Mapa earliest/latest por ticker
    earliest = cache_earliest_date_per_ticker(all_tickers)
    latest   = cache_latest_date_per_ticker(all_tickers)

    # -------- BACKFILL (si la ventana pide fechas más antiguas que earliest) --------
    need_back = []
    back_start_map = {}
    if start_date is not None:
        sdt = pd.to_datetime(start_date).normalize()
        for t in all_tickers:
            e0 = earliest.get(t)
            if e0 is None or (not pd.isna(e0) and sdt < e0):
                need_back.append(t)
                # si no hay earliest, pedimos 10 años hacia atrás desde sdt
                back_start_map[t] = start_date
        if need_back:
            end_map = {}
            for t in need_back:
                e0 = earliest.get(t)
                if e0 is None:
                    # bajamos desde start_date hasta hoy (luego forward acota)
                    end_map[t] = datetime.utcnow().strftime("%Y-%m-%d")
                else:
                    end_map[t] = (pd.to_datetime(e0) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            back_df = fetch_backfill_yf(need_back, start_date, min(end_map.values()))
            if back_df is not None and not back_df.empty:
                cache_append_prices(back_df)

    # -------- FORWARD (si falta desde la última fecha en cache hasta hoy) --------
    today = pd.Timestamp(datetime.utcnow().date())
    needs = []
    start_map = {}
    for t in all_tickers:
        ld = latest.get(t)
        if ld is None or pd.Timestamp(ld).normalize() < today:
            needs.append(t)
            start_map[t] = start_date if ld is None else pd.to_datetime(ld).strftime("%Y-%m-%d")
    if needs:
        missing_df = fetch_from_yf_missing(needs, start_map, end_date=None)
        if missing_df is not None and not missing_df.empty:
            cache_append_prices(missing_df)

    # 4) consolidado del cache (ya actualizado)
    consolidated = cache_read_prices(all_tickers, start_date=start_date)

    # Fallback directo si aún no hay nada
    failed_all = []; errs_all = []
    if consolidated is None or consolidated.empty:
        from_direct, failed_d, errs_d = fetch_prices_yahoo_direct(all_tickers, start=start_date, end=None)
        if not from_direct.empty:
            cache_append_prices(from_direct)
            consolidated = from_direct
        failed_all = failed_d; errs_all += errs_d

    bench_ret = pd.Series(dtype=float)
    prices = consolidated.copy()
    if not prices.empty and bench in prices.columns:
        bench_ret = prices[[bench]].pct_change().dropna()[bench]
        prices = prices.drop(columns=[bench], errors="ignore")

    return prices, bench_ret, (failed_all or []), (errs_all or [])

# ================== MI PORTAFOLIO ==================
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")
    if st.button("🔄 Refrescar datos"): refresh_all(); st.rerun()

    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("No hay posiciones. Carga operaciones en 'Transactions'."); st.stop()

    tickers = pos_df["Ticker"].tolist()
    prices, bench_ret, failed_all, errs_all = load_prices_with_fallback(tickers, benchmark, start_date)

    if prices.empty or len(prices.columns)==0:
        st.warning("No se pudieron obtener precios históricos.")
        if failed_all: st.caption("Fallidos: " + ", ".join(sorted(set(failed_all))))
        st.stop()

    # Salvaguarda: si hay menos de 2 días, avisamos (evita métricas y gráfico planos)
    if prices.shape[0] < 2:
        st.info("Necesito al menos 2 días de histórico para calcular métricas. Se está completando el cache; vuelve a intentarlo en segundos.")
        # seguimos mostrando la tabla de posiciones:
    w = weights_from_positions(pos_df)

    # Cálculos de tabla
    since_buy = (pos_df.set_index("Ticker")["MarketPrice"]/pos_df.set_index("Ticker")["AvgCost"] - 1).replace([np.inf,-np.inf], np.nan)
    window_change = (prices.ffill().iloc[-1]/prices.ffill().iloc[0]-1).reindex(pos_df["Ticker"]).fillna(np.nan)
    pos_df = pos_df.set_index("Ticker").loc[w.index].reset_index()

    # --------- TABLA (sin columna "#") + columna "➖" para eliminar ----------
    view = pd.DataFrame({
        "Ticker": pos_df["Ticker"].values,
        "Shares": pos_df["Shares"].values,
        "Avg Buy": pos_df["AvgCost"].values,
        "Last": pos_df["MarketPrice"].values,
        "P/L $": pos_df["UnrealizedPL"].values,
        "P/L % (compra)": (since_buy.reindex(pos_df["Ticker"]).values*100.0),
        "Δ % ventana": (window_change.reindex(pos_df["Ticker"]).values*100.0),
        "Peso %": (w.reindex(pos_df["Ticker"]).values*100.0),
        "Valor": pos_df["MarketValue"].values,
        "➖": [False]*len(pos_df)
    }).replace([np.inf,-np.inf], np.nan)

    colcfg = {
        "➖": st.column_config.CheckboxColumn(
            label="➖", help="Marcar para eliminar este ticker de Transactions",
            width="small", default=False
        )
    }

    edited = st.data_editor(
        view, hide_index=True, use_container_width=True,
        column_config=colcfg,
        disabled=[c for c in view.columns if c!="➖"],
        key="positions_editor"
    )

    # Detectar click (marcado nuevo)
    prev = st.session_state.get("prev_editor_df")
    if prev is None:
        st.session_state["prev_editor_df"] = edited.copy()
    else:
        mark = (edited["➖"] == True) & (prev["➖"] != True)
        if mark.any():
            row_idx = mark[mark].index[0]
            st.session_state["delete_candidate"] = str(edited.iloc[row_idx]["Ticker"])
        st.session_state["prev_editor_df"] = edited.copy()

    # Confirmación
    if st.session_state.get("delete_candidate"):
        tkr = st.session_state["delete_candidate"]
        st.warning(f"¿Seguro que quieres eliminar **todas** las transacciones de **{tkr}** en *Transactions*?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Sí, eliminar"):
                deleted = delete_transactions_by_ticker(tkr)
                st.session_state["delete_candidate"] = ""
                st.session_state["prev_editor_df"]["➖"] = False
                if deleted>0:
                    st.success(f"Se eliminaron {deleted} fila(s) de {tkr} en Transactions.")
                else:
                    st.info("No se encontraron filas para eliminar.")
                refresh_all(); st.rerun()
        with c2:
            if st.button("❌ No, cancelar"):
                st.session_state["delete_candidate"] = ""
                st.session_state["prev_editor_df"]["➖"] = False
                st.info("Operación cancelada.")

    # ---- KPIs y gráfico (solo si hay ≥ 2 días) ----
    if prices.shape[0] >= 2:
        c_top1, c_top2 = st.columns([2,1])
        with c_top1:
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

# ================== OPTIMIZAR ==================
elif page=="Optimizar y Rebalancear":
    st.title("🛠️ Optimizar y Rebalancear")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("No hay posiciones."); st.stop()
    tickers = pos_df["Ticker"].tolist()
    prices, bench_ret, failed_all, errs_all = load_prices_with_fallback(tickers, benchmark, start_date)
    if prices.empty: st.warning("No hay precios para optimizar."); st.stop()
    if failed_all:
        st.caption("⚠️ Excluidos por falta de histórico: " + ", ".join(sorted(set(failed_all))))
        good = [c for c in prices.columns if prices[c].notna().any()]
        prices = prices[good]; pos_df = pos_df[pos_df["Ticker"].isin(good)].copy()

    w_cur = weights_from_positions(pos_df)
    port_ret_cur, asset_rets = const_weight_returns(prices, w_cur)
    if asset_rets.empty: st.warning("No hay retornos suficientes para optimizar."); st.stop()

    mean_daily=asset_rets.mean(); cov=asset_rets.cov(); mu_ann=(1+mean_daily)**252-1
    n=len(mu_ann); bounds=[(w_min,w_max)]*n
    w_opt_arr = max_sharpe(mu_ann.values, cov.values, rf=rf, bounds=bounds)
    w_opt = pd.Series(w_opt_arr, index=mu_ann.index).clip(lower=w_min, upper=w_max); w_opt /= w_opt.sum()

    port_ret_opt,_ = const_weight_returns(prices, w_opt)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sharpe actual", f"{(sharpe(port_ret_cur, rf) or 0):.2f}")
    c2.metric("Sharpe propuesto", f"{(sharpe(port_ret_opt, rf) or 0):.2f}")
    c3.metric("Vol. actual", f"{(annualize_vol(port_ret_cur) or 0)*100:,.2f}%")
    c4.metric("Vol. propuesto", f"{(annualize_vol(port_ret_opt, rf) or 0)*100:,.2f}%")
    compare=pd.DataFrame({"Weight Actual":w_cur,"Weight Propuesto":w_opt}).fillna(0)
    compare["Δ (pp)"]=(compare["Weight Propuesto"]-compare["Weight Actual"])*100
    st.data_editor(compare, hide_index=False, use_container_width=True,
                   column_config={"Weight Actual":st.column_config.NumberColumn(format="%.2f%%"),
                                  "Weight Propuesto":st.column_config.NumberColumn(format="%.2f%%"),
                                  "Δ (pp)":st.column_config.NumberColumn(format="%.2f")},
                   disabled=list(compare.columns))

# ================== EVALUAR CANDIDATO ==================
elif page=="Evaluar Candidato":
    st.title("🧪 Evaluar Candidato")
    pos_df = positions_from_tx(tx_df)
    if pos_df.empty: st.info("Sin posiciones."); st.stop()
    tkr = st.text_input("Ticker a evaluar", value="AAPL").upper().strip().replace(" ","")
    if not tkr: st.stop()
    tickers = sorted(set(pos_df["Ticker"].tolist()+[tkr]))
    prices, bench_ret, failed_all, errs_all = load_prices_with_fallback(tickers, benchmark, start_date)
    if prices.empty: st.warning("No se pudieron descargar precios."); st.stop()

    w_cur = weights_from_positions(pos_df)
    last_map, fail_lp = last_prices_yf_safe([tkr])
    if fail_lp:
        last_map2, _ = last_prices_direct(fail_lp); last_map.update(last_map2)
    last = last_map.get(tkr, np.nan)
    if np.isnan(last): st.warning("Ticker inválido o sin último precio."); st.stop()

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
    c[3].metric("MDD con candidato", f"{(0 if pd.isna(mdd_new) else mdd_new)*100:,.2f}%")

# ================== EXPLORAR / DIAGNÓSTICO ==================
elif page=="Explorar / Research":
    st.title("🔎 Explorar / Research")
    tkr=st.text_input("Ticker", value="MSFT").upper().strip().replace(" ","")
    if tkr:
        hist, bench_ret, failed_all, errs_all = load_prices_with_fallback([tkr], benchmark, start_date)
        if not hist.empty and tkr in hist.columns:
            ser = hist[tkr].dropna()
            if not ser.empty:
                df = pd.DataFrame({"Close": ser})
                fig=go.Figure(data=[go.Candlestick(
                    x=df.index, open=df["Close"], high=df["Close"], low=df["Close"], close=df["Close"]
                )])
                fig.update_layout(title=f"Serie (Close) {tkr}", xaxis_rangeslider_visible=False, height=480)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Yahoo no entregó histórico para ese ticker.")
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
        st.subheader("Ping yfinance (SPY 30d)")
        try:
            test = yf.download("SPY", period="30d", interval="1d", auto_adjust=True,
                               progress=False, threads=False)
            if test is None or test.empty:
                st.warning("Sin datos por yfinance (rate-limit o red).")
            else:
                st.success(f"OK YF: {len(test)} barras.")
                st.dataframe(test.tail(5))
        except Exception as e:
            st.error(f"Error yfinance: {e}")
    st.subheader("Transactions (muestra)")
    st.dataframe(load_all_data()[0].head(10), use_container_width=True)
