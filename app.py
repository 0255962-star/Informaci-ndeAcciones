# APP Finanzas – Portafolio con cache incremental + backfill en Google Sheets
from datetime import datetime, timedelta, timezone
import os, time, logging
import numpy as np
import pandas as pd
import plotly.express as px
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
:root { --bg:#0b0f1a; --card:#141927; --muted:#8b93a7; --brand:#7c6cff; --up:#18b26b; --down:#ff4d4f; }
.block-container{padding-top:.8rem}
section[data-testid="stSidebar"]{border-right:1px solid #1e2435}
div[data-testid="stMetricValue"]{font-size:1.6rem}
[data-testid="stHeader"]{background:rgba(0,0,0,0)}
th:has(> div:contains("➖")){color:#ff4d4f !important; width:48px !important}
td:has(input[type="checkbox"]){text-align:center}
</style>
""", unsafe_allow_html=True)

# Silenciar logger ruidoso de yfinance
try:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
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

# ================== UTILIDADES ==================
def refresh_all():
    for f in (read_sheet, load_all_data, cache_read_prices, cache_latest_date_per_ticker,
              cache_earliest_date_per_ticker, last_prices_from_cache, fetch_prices_yahoo_direct,
              last_prices_yf_safe, last_prices_direct, load_prices_with_fallback):
        try: f.clear()
        except Exception: pass
    try: st.cache_data.clear()
    except Exception: pass

def _clean_tickers(tickers):
    out=[]
    for t in tickers:
        if not isinstance(t,str): continue
        t=t.upper().strip().replace(" ","")
        if t and t not in out: out.append(t)
    return out

# ================== LECTURA DE SHEETS ==================
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

# ================== CACHE DE PRECIOS EN SHEETS ==================
CACHE_SHEET_NAME = "PricesCache"   # A:Date, B:Ticker, C:Close

def _ensure_cache_sheet():
    sh = gc.open_by_key(SHEET_ID)
    if CACHE_SHEET_NAME not in [w.title for w in sh.worksheets()]:
        ws = sh.add_worksheet(title=CACHE_SHEET_NAME, rows=1000, cols=3)
        ws.update("A1:C1", [["Date","Ticker","Close"]])

def _get_cache_ws():
    _ensure_cache_sheet()
    return gc.open_by_key(SHEET_ID).worksheet(CACHE_SHEET_NAME)

@st.cache_data(ttl=0, show_spinner=False)
def cache_read_prices(tickers, start_date=None):
    ws = _get_cache_ws()
    values = ws.get_all_values()
    if not values or len(values) < 2: return pd.DataFrame()
    df = pd.DataFrame(values[1:], columns=values[0])
    df = df[(df["Date"]!="") & (df["Ticker"]!="") & (df["Close"]!="")]
    if df.empty: return pd.DataFrame()
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Ticker","Close"])
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date).normalize()]
    if tickers:
        df = df[df["Ticker"].isin(set(_clean_tickers(tickers)))]
    if df.empty: return pd.DataFrame()
    wide = df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()
    return wide[~wide.index.duplicated(keep="last")]

@st.cache_data(ttl=0, show_spinner=False)
def cache_latest_date_per_ticker(tickers):
    ws = _get_cache_ws(); values = ws.get_all_values()
    out = {t: None for t in tickers}
    if not values or len(values) < 2: return out
    df = pd.DataFrame(values[1:], columns=values[0])
    df = df[(df["Date"]!="") & (df["Ticker"]!="")]
    if df.empty: return out
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    for t in _clean_tickers(tickers):
        sub = df.loc[df["Ticker"]==t, "Date"]
        out[t] = sub.max() if not sub.empty else None
    return out

@st.cache_data(ttl=0, show_spinner=False)
def cache_earliest_date_per_ticker(tickers):
    ws = _get_cache_ws(); values = ws.get_all_values()
    out = {t: None for t in tickers}
    if not values or len(values) < 2: return out
    df = pd.DataFrame(values[1:], columns=values[0])
    df = df[(df["Date"]!="") & (df["Ticker"]!="")]
    if df.empty: return out
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    for t in _clean_tickers(tickers):
        sub = df.loc[df["Ticker"]==t, "Date"]
        out[t] = sub.min() if not sub.empty else None
    return out

def cache_append_prices(df_wide):
    if df_wide is None or df_wide.empty: return 0
    ws = _get_cache_ws()
    df_wide = df_wide.copy()
    df_wide.index = pd.to_datetime(df_wide.index).normalize()
    existing = cache_read_prices(list(df_wide.columns))
    to_write = df_wide.copy()
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, to_write]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        # escribir sólo lo nuevo
        mask_new = merged.loc[to_write.index, to_write.columns].where(
            ~existing.reindex(merged.index).notna(), other=np.nan)
        to_write = mask_new.dropna(how="all")
    if to_write.empty: return 0
    rows = []
    for dt, row in to_write.sort_index().iterrows():
        for t, val in row.items():
            if pd.isna(val): continue
            rows.append([dt.strftime("%Y-%m-%d"), str(t), float(val)])
    total = 0; BATCH = 800
    for i in range(0, len(rows), BATCH):
        ws.append_rows(rows[i:i+BATCH], value_input_option="RAW")
        total += len(rows[i:i+BATCH])
    return total

# ================== PRECIOS: YF / DIRECTO ==================
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
    # Preferimos adjclose; si no, close
    try:
        adj = r.get("indicators", {}).get("adjclose", [])
        if adj and "adjclose" in adj[0]:
            vals = adj[0]["adjclose"]; return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception: pass
    try:
        quote = r.get("indicators", {}).get("quote", [])
        if quote and "close" in quote[0]:
            vals = quote[0]["close"];  return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception: pass
    return pd.Series(dtype=float)

def _direct_one(ticker, start=None, end=None, interval="1d", timeout=25):
    params = {"interval": interval, "includeAdjustedClose": "true", "events": "div,splits"}
    if start is None and end is None: params["range"] = "max"
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
                    last_err = f"HTTP {r.status_code}"; time.sleep(0.6*(k+1)); continue
                s = _parse_chart_json(r.json())
                if isinstance(s, pd.Series) and not s.empty and len(s.dropna()) >= 3:
                    s.name = ticker; return s, None
                last_err = "respuesta vacía"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.6*(k+1))
    return pd.Series(dtype=float), last_err

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices_yahoo_direct(tickers, start=None, end=None, interval="1d"):
    tickers = _clean_tickers(tickers)
    if not tickers: return pd.DataFrame(), [], []
    frames=[]; failed=[]; errs=[]
    for t in tickers:
        s, err = _direct_one(t, start=start, end=end, interval=interval)
        if isinstance(s, pd.Series) and not s.empty: frames.append(s.rename(t).to_frame())
        else:
            failed.append(t); 
            if err: errs.append(f"[direct:{t}] {err}")
    if frames:
        df = pd.concat(frames, axis=1).sort_index()
        df.index = pd.to_datetime(df.index).normalize()
        return df, failed, errs
    return pd.DataFrame(), tickers, errs

@st.cache_data(ttl=1500, show_spinner=False)
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

@st.cache_data(ttl=1200, show_spinner=False)
def last_prices_direct(tickers):
    tickers = _clean_tickers(tickers)
    out, failed = {}, []
    for t in tickers:
        s, _ = _direct_one(t, interval="1d")
        if isinstance(s, pd.Series) and not s.empty:
            out[t] = float(s.dropna().iloc[-1])
        else:
            failed.append(t)
    return out, failed

# ================== CARGA / SINCRONIZACIÓN DE HISTÓRICO ==================
def _single_yf_range(ticker, start, end):
    try:
        d = yf.download(ticker, start=start, end=end, interval="1d",
                        auto_adjust=True, progress=False, threads=False)
        if d is None or d.empty: return pd.Series(dtype=float)
        s = pd.Series(d["Close"]).dropna(); s.index = pd.to_datetime(s.index).normalize()
        s.name = ticker; return s
    except Exception:
        return pd.Series(dtype=float)

def load_prices_with_fallback(tickers, bench, start_date):
    """1) Leer cache. 2) Backfill por ticker si la ventana va más atrás.
       3) Forward-fill (añadir sólo días hábiles nuevos). 4) Releer cache consolidado.
       5) Si todo falla, usar Yahoo directo."""
    all_tickers = list(dict.fromkeys((tickers or []) + [bench]))
    all_tickers = _clean_tickers(all_tickers)

    # ——— BACKFILL (si la ventana arranca antes de lo que hay en cache)
    earliest = cache_earliest_date_per_ticker(all_tickers)
    if start_date is not None:
        sdt = pd.to_datetime(start_date).normalize()
        for t in all_tickers:
            e0 = earliest.get(t)
            if e0 is None or (pd.notna(e0) and sdt < e0):
                # traer tramo [start_date, e0 - 1d]
                end_dt = (pd.to_datetime(e0).normalize() - pd.Timedelta(days=1)) if e0 is not None else None
                s = _single_yf_range(t, start=start_date, end=end_dt)
                if not s.empty: cache_append_prices(s.to_frame())

    # ——— FORWARD (desde la última fecha hasta hoy)
    latest = cache_latest_date_per_ticker(all_tickers)
    today = pd.Timestamp(datetime.utcnow().date())
    for t in all_tickers:
        ld = latest.get(t)
        # si nunca hubo datos, trae 3 años por default
        if ld is None:
            base_start = (today - pd.Timedelta(days=365*3)).strftime("%Y-%m-%d")
            s = _single_yf_range(t, start=base_start, end=None)
            if s.empty:
                # último intento: directo
                s2, _ = _direct_one(t, start=base_start, end=None)
                if not s2.empty: cache_append_prices(s2.to_frame())
            else:
                cache_append_prices(s.to_frame())
        else:
            if pd.Timestamp(ld).normalize() < today:
                start = (pd.to_datetime(ld).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                s = _single_yf_range(t, start=start, end=None)
                if s.empty:
                    s2, _ = _direct_one(t, start=start, end=None)
                    if not s2.empty: cache_append_prices(s2.to_frame())
                else:
                    cache_append_prices(s.to_frame())

    # ——— Consolidado final para la ventana solicitada
    consolidated = cache_read_prices(all_tickers, start_date=start_date)

    # Si por alguna razón quedó vacío, intentar Yahoo direct para todo y escribirlo
    if consolidated is None or consolidated.empty:
        from_direct, _, _ = fetch_prices_yahoo_direct(all_tickers, start=start_date, end=None)
        if not from_direct.empty:
            cache_append_prices(from_direct)
            consolidated = from_direct

    # separa benchmark
    bench_ret = pd.Series(dtype=float)
    prices = consolidated.copy()
    if not prices.empty and bench in prices.columns:
        bench_ret = prices[[bench]].pct_change().dropna()[bench]
        prices = prices.drop(columns=[bench], errors="ignore")
    return prices, bench_ret

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

def positions_from_tx(tx:pd.DataFrame, last_hint_map:dict|None=None):
    if tx.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    df=tidy_transactions(tx)
    uniq=sorted(df["Ticker"].unique().tolist())
    if not uniq:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])

    # 1) usar “hint” del histórico y, si falta, usar cache global (sin ventana)
    last_map = dict(last_hint_map or {})
    missing = [t for t in uniq if t not in last_map or pd.isna(last_map.get(t))]
    if missing:
        cached_fallback = last_prices_from_cache(missing)  # del cache completo
        last_map.update(cached_fallback)
        missing = [t for t in missing if t not in last_map or pd.isna(last_map.get(t))]
    # 2) si aún faltan, yfinance
    if missing:
        lm2, _ = last_prices_yf_safe(missing); last_map.update(lm2)
        missing = [t for t in missing if t not in last_map or pd.isna(last_map.get(t))]
    # 3) si aún faltan, directo
    if missing:
        lm3, _ = last_prices_direct(missing); last_map.update(lm3)

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

        px = last_map.get(t, np.nan)
        mv = sh*px if not np.isnan(px) else np.nan
        inv= sh*avg if not np.isnan(avg) else np.nan
        pl = mv-inv if not (np.isnan(mv) or np.isnan(inv)) else np.nan
        pos.append([t,sh,avg,inv,px,mv,pl])

    dfp=pd.DataFrame(pos,columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    return dfp.sort_values("MarketValue",ascending=False)

@st.cache_data(ttl=60, show_spinner=False)
def last_prices_from_cache(tickers):
    """Regresa {ticker: ultimo_close} usando todo el cache (sin filtrar por ventana)."""
    tickers = _clean_tickers(tickers)
    if not tickers: return {}
    df = cache_read_prices(tickers, start_date=None)
    if df is None or df.empty: return {}
    out={}
    for t in tickers:
        try:
            s = df[t].dropna()
            if not s.empty: out[t] = float(s.iloc[-1])
        except Exception: pass
    return out

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
    price_df=price_df[cols].dropna(how="all").ffill()
    rets=price_df.pct_change().dropna(how="all")
    if rets.empty: return pd.Series(dtype=float), pd.DataFrame()
    w=weights.reindex(rets.columns).fillna(0).values
    port=(rets*w).sum(axis=1)
    return port, rets

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
    for ridx in reversed(to_delete): ws.delete_rows(ridx)
    return len(to_delete)

# ================== CONFIG / SIDEBAR ==================
tx_df, settings_df, _ = load_all_data()
rf = get_setting(settings_df,"RF",0.03,float)
benchmark = get_setting(settings_df,"Benchmark","SPY",str)

st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Diagnóstico"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ================== HELPER PARA ALINEAR TABLA VS PESOS ==================
def align_positions_for_view(positions_df: pd.DataFrame, weights_index: pd.Index) -> pd.DataFrame:
    if positions_df.empty: return positions_df
    tmp = positions_df.set_index("Ticker")
    if weights_index is not None and len(weights_index) > 0:
        tmp = tmp.reindex(weights_index)
    out = tmp.reset_index()
    if "Ticker" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "Ticker"})
    return out

# ================== MI PORTAFOLIO ==================
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")
    if st.button("🔄 Refrescar datos"):
        refresh_all(); st.rerun()

    # Tickers desde Transactions
    tx_tickers = sorted(set(
        t for t in tx_df.get("Ticker", pd.Series(dtype=str)).astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
        if t
    ))
    if not tx_tickers:
        st.info("No hay posiciones. Carga operaciones en 'Transactions'."); st.stop()

    # 1) Sincroniza histórico (lee de cache, hace backfill/forward por ticker, y vuelve a cache)
    prices, bench_ret = load_prices_with_fallback(tx_tickers, benchmark, start_date)

    # 2) Mapa de últimos precios: usa precios de la ventana y, si falta, cache global
    last_hint_map = {}
    if prices is not None and not prices.empty:
        last_hint_map.update(prices.ffill().iloc[-1].dropna().astype(float).to_dict())
    # fallback a cache completo para cualquier ticker sin valor
    missing_for_last = [t for t in tx_tickers if t not in last_hint_map or pd.isna(last_hint_map.get(t))]
    if missing_for_last:
        last_hint_map.update(last_prices_from_cache(missing_for_last))

    # 3) Construir posiciones con ese mapa (sin cambios de UI)
    pos_df = positions_from_tx(tx_df, last_hint_map=last_hint_map)
    if pos_df.empty:
        st.info("No hay posiciones válidas tras procesar Transactions."); st.stop()

    # 4) Pesos y tabla
    w = weights_from_positions(pos_df)
    pos_df_view = align_positions_for_view(pos_df, w.index)

    # métricas de columnas calculadas
    since_buy = (pos_df.set_index("Ticker")["MarketPrice"]/pos_df.set_index("Ticker")["AvgCost"] - 1)\
                .replace([np.inf,-np.inf], np.nan)
    window_change = pd.Series(index=pos_df_view["Ticker"], dtype=float)
    if prices is not None and not prices.empty and prices.shape[0] >= 2:
        window_change = (prices.ffill().iloc[-1]/prices.ffill().iloc[0]-1)\
                        .reindex(pos_df_view["Ticker"]).fillna(np.nan)

    # ——— Tabla (misma estructura visual)
    view = pd.DataFrame({
        "Ticker": pos_df_view["Ticker"].values,
        "Shares": pos_df_view["Shares"].values,
        "Avg Buy": pos_df_view["AvgCost"].values,
        "Last": pos_df_view["MarketPrice"].values,
        "P/L $": pos_df_view["UnrealizedPL"].values,
        "P/L % (compra)": (since_buy.reindex(pos_df_view["Ticker"]).values*100.0),
        "Δ % ventana": (window_change.reindex(pos_df_view["Ticker"]).values*100.0),
        "Peso %": (w.reindex(pos_df_view["Ticker"]).values*100.0),
        "Valor": pos_df_view["MarketValue"].values,
        "➖": [False]*len(pos_df_view)
    }).replace([np.inf,-np.inf], np.nan)

    colcfg = {"➖": st.column_config.CheckboxColumn(label="➖", help="Marcar para eliminar este ticker", width="small", default=False)}
    # Key estable por ventana para que no “salte” al cambiar de pestaña
    editor_key = f"positions_editor_{window}"
    edited = st.data_editor(view, hide_index=True, use_container_width=True,
                            column_config=colcfg, disabled=[c for c in view.columns if c!="➖"],
                            key=editor_key)

    # control de selección de borrado
    prev_key = f"prev_editor_df_{window}"
    prev = st.session_state.get(prev_key)
    if prev is None:
        st.session_state[prev_key] = edited.copy()
    else:
        mark = (edited["➖"] == True) & (prev["➖"] != True)
        if mark.any():
            row_idx = mark[mark].index[0]
            st.session_state["delete_candidate"] = str(edited.iloc[row_idx]["Ticker"])
        st.session_state[prev_key] = edited.copy()

    if st.session_state.get("delete_candidate"):
        tkr = st.session_state["delete_candidate"]
        st.warning(f"¿Seguro que quieres eliminar **todas** las transacciones de **{tkr}** en *Transactions*?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Sí, eliminar"):
                deleted = delete_transactions_by_ticker(tkr)
                st.session_state["delete_candidate"] = ""
                st.session_state[prev_key]["➖"] = False
                if deleted>0: st.success(f"Se eliminaron {deleted} fila(s) de {tkr}.")
                else: st.info("No se encontraron filas para eliminar.")
                refresh_all(); st.rerun()
        with c2:
            if st.button("❌ No, cancelar"):
                st.session_state["delete_candidate"] = ""
                st.session_state[prev_key]["➖"] = False
                st.info("Operación cancelada.")

    # ——— KPIs y gráficos (idéntico layout)
    if prices is not None and not prices.empty and prices.shape[0] >= 2 and len(w)>0:
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
            if bench_ret is not None and not bench_ret.empty:
                curve["Benchmark"]=(1+bench_ret).cumprod().reindex(curve.index).ffill()
            st.plotly_chart(px.line(curve, title="Crecimiento de 1.0"), use_container_width=True)
        with c_top2:
            alloc = pd.DataFrame({"Ticker":w.index,"Weight":w.values})
            st.plotly_chart(px.pie(alloc, names="Ticker", values="Weight", title="Asignación"), use_container_width=True)
    elif len(w)==0:
        st.info("No hay pesos válidos para calcular métricas.")
    else:
        st.info("Necesito al menos 2 días de histórico para calcular métricas.")

# (las demás páginas permanecen tal cual)
