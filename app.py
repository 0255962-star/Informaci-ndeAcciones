# APP Finanzas – Portafolio con histórico en Google Sheets
# Optimización: masters por sesión, sync sólo en "Refrescar", cache estable y UI sin cambios.

from datetime import datetime, timedelta, timezone, date
import os, time, logging, re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import yfinance as yf
from scipy.optimize import minimize
import gspread
from google.oauth2.service_account import Credentials

# ================== CONFIG UI ==================
st.set_page_config(page_title="APP Finanzas – Portafolio Activo", page_icon="💼", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:.8rem}
section[data-testid="stSidebar"]{border-right:1px solid #1e2435}
div[data-testid="stMetricValue"]{font-size:1.6rem}
th:has(> div:contains("➖")){color:#ff4d4f !important; width:48px !important}
td:has(input[type="checkbox"]){text-align:center}
</style>
""", unsafe_allow_html=True)

try:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    yf.utils.get_yf_logger().setLevel(logging.CRITICAL)
except Exception:
    pass

# ================== SECRETS ==================
SHEET_ID = st.secrets.get("SHEET_ID") or st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
GCP_SA   = st.secrets.get("gcp_service_account", {})
if not SHEET_ID: st.error("Falta `SHEET_ID` en secrets."); st.stop()
if not GCP_SA:   st.error("Falta `gcp_service_account` en secrets."); st.stop()

# ================== CACHE DE RECURSOS (no se recrea el cliente) ==================
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(GCP_SA, scopes=scopes)
    return gspread.authorize(credentials)

gc = get_gspread_client()

# ================== UTILS ==================
def _clean_tickers(tickers):
    out=[]
    for t in tickers:
        if not isinstance(t,str): continue
        t=t.upper().strip().replace(" ","")
        if t and t not in out: out.append(t)
    return out

def _is_dtlike(x):
    return isinstance(x, (pd.Timestamp, datetime, date))

def _parse_number(x):
    """Locale-aware: detecta último separador decimal (coma/punto) y normaliza a float."""
    if x is None: return np.nan
    if isinstance(x,(int,float)): return float(x)
    if _is_dtlike(x): return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    s = re.sub(r"[^\d,\.\-\s]", "", s).replace(" ", "")
    if "," in s and "." in s:
        last_comma = s.rfind(","); last_dot = s.rfind(".")
        if last_comma > last_dot:  # coma decimal
            s = s.replace(".", "").replace(",", ".")
        else:                      # punto decimal
            s = s.replace(",", "")
    else:
        if "," in s: s = s.replace(",", ".")
    if s in ("", ".", "-", "-.", ".-"): return np.nan
    try: return float(s)
    except: return np.nan

# ================== SHEETS HELPERS ==================
def open_ws(name:str):
    return gc.open_by_key(SHEET_ID).worksheet(name)

@st.cache_data(ttl=1200, show_spinner=False)
def read_sheet(name:str)->pd.DataFrame:
    ws = open_ws(name)
    return pd.DataFrame(ws.get_all_records())

def get_setting(settings_df, key, default=None, cast=float):
    try:
        s = settings_df.loc[settings_df["Key"]==key,"Value"]
        if s.empty: return default
        return cast(s.values[0]) if cast else s.values[0]
    except Exception:
        return default

# ======== Hoja de cache de precios (autodetección una sóla vez) ========
CACHE_CANON = "PricesCache"

def _norm_cols(cols):
    return [str(c).strip().lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u") for c in cols]

@st.cache_resource(show_spinner=False)
def find_cache_meta():
    sh = gc.open_by_key(SHEET_ID)
    candidatas = []
    for ws in sh.worksheets():
        vals = ws.get_all_values()
        if not vals or len(vals)<1: continue
        header = _norm_cols(vals[0])
        cols = {c:i for i,c in enumerate(header)}
        date_keys   = [k for k in ("date","fecha") if k in cols]
        ticker_keys = [k for k in ("ticker","symbol","simbolo") if k in cols]
        close_keys  = [k for k in ("close","cierre","adj close","adj_close","adjclose") if k in cols]
        score = int(len(date_keys)>0) + int(len(ticker_keys)>0) + int(len(close_keys)>0)
        if score >= 2:
            candidatas.append((ws.title, date_keys[:1], ticker_keys[:1], close_keys[:1]))
    for c in candidatas:
        if c[0].strip().lower()==CACHE_CANON.lower():
            return {"sheet": c[0], "date_col": c[1][0] if c[1] else "date",
                    "ticker_col": c[2][0] if c[2] else "ticker",
                    "close_col": c[3][0] if c[3] else "close"}
    if candidatas:
        c=candidatas[0]
        return {"sheet": c[0], "date_col": c[1][0] if c[1] else "date",
                "ticker_col": c[2][0] if c[2] else "ticker",
                "close_col": c[3][0] if c[3] else "close"}
    ws = gc.open_by_key(SHEET_ID).add_worksheet(title=CACHE_CANON, rows=1000, cols=3)
    ws.update("A1:C1", [["Date","Ticker","Close"]])
    return {"sheet": CACHE_CANON, "date_col":"date", "ticker_col":"ticker", "close_col":"close"}

def _get_cache_ws():
    meta = find_cache_meta()
    return open_ws(meta["sheet"]), meta

@st.cache_data(ttl=1200, show_spinner=False)
def cache_read_prices(tickers, start_date=None):
    ws, meta = _get_cache_ws()
    values = ws.get_all_values()
    if not values or len(values) < 2: return pd.DataFrame()
    header = _norm_cols(values[0])
    df = pd.DataFrame(values[1:], columns=header)
    dcol = meta["date_col"]; tcol = meta["ticker_col"]; ccol = meta["close_col"]
    for col in (dcol, tcol, ccol):
        if col not in df.columns: return pd.DataFrame()
    df = df[(df[dcol]!="") & (df[tcol]!="") & (df[ccol]!="")]
    if df.empty: return pd.DataFrame()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    df[tcol] = df[tcol].astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
    df[ccol] = df[ccol].map(_parse_number)
    df = df.dropna(subset=[dcol,tcol,ccol])
    if start_date:
        df = df[df[dcol] >= pd.to_datetime(start_date).normalize()]
    if tickers:
        df = df[df[tcol].isin(set(_clean_tickers(tickers)))]
    if df.empty: return pd.DataFrame()
    wide = df.pivot_table(index=dcol, columns=tcol, values=ccol, aggfunc="last").sort_index()
    return wide[~wide.index.duplicated(keep="last")]

def cache_append_prices(df_wide):
    if df_wide is None or df_wide.empty: return 0
    ws, meta = _get_cache_ws()
    df_wide = df_wide.copy()
    df_wide.index = pd.to_datetime(df_wide.index).normalize()

    existing = cache_read_prices(list(df_wide.columns))
    to_write = df_wide.copy()
    if existing is not None and not existing.empty:
        merged = pd.concat([existing, to_write]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        mask_new = merged.loc[to_write.index, to_write.columns].where(
            ~existing.reindex(merged.index).notna(), other=np.nan)
        to_write = mask_new.dropna(how="all")
    if to_write.empty: return 0

    header_real = ws.get_all_values()[0]
    header_norm = _norm_cols(header_real)
    idx = {c:i for i,c in enumerate(header_norm)}
    dcol = meta["date_col"]; tcol = meta["ticker_col"]; ccol = meta["close_col"]
    if dcol not in idx or tcol not in idx or ccol not in idx:
        ws.update("A1:C1", [["Date","Ticker","Close"]])
        header_real = ["Date","Ticker","Close"]
        header_norm = _norm_cols(header_real)
        idx = {c:i for i,c in enumerate(header_norm)}
        dcol, tcol, ccol = "date","ticker","close"

    rows=[]
    for dt_i, row in to_write.sort_index().iterrows():
        for t, val in row.items():
            if pd.isna(val): continue
            r = [""] * len(header_real)
            r[idx[dcol]] = pd.Timestamp(dt_i).strftime("%Y-%m-%d")
            r[idx[tcol]] = str(t)
            r[idx[ccol]] = float(val)
            rows.append(r)

    BATCH=800; total=0
    for i in range(0, len(rows), BATCH):
        ws.append_rows(rows[i:i+BATCH], value_input_option="RAW")
        total += len(rows[i:i+BATCH])
    return total

# ================== DESCARGA PRECIOS (fallback Yahoo) ==================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
os.environ.setdefault("YF_USER_AGENT", UA)

def _unix(dt_): return int(pd.Timestamp(dt_, tz=timezone.utc).timestamp())

def _parse_chart_json(js) -> pd.Series:
    result = js.get("chart", {}).get("result", [])
    if not result: return pd.Series(dtype=float)
    r = result[0]; ts = r.get("timestamp", [])
    if not ts: return pd.Series(dtype=float)
    idx = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert(None).normalize()
    try:
        adj = r.get("indicators", {}).get("adjclose", [])
        if adj and "adjclose" in adj[0]:
            vals = adj[0]["adjclose"]; return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception: pass
    try:
        q = r.get("indicators", {}).get("quote", [])
        if q and "close" in q[0]:
            vals = q[0]["close"]; return pd.Series(vals, index=idx, dtype="float64").dropna()
    except Exception: pass
    return pd.Series(dtype=float)

def _direct_one(ticker, start=None, end=None, interval="1d", timeout=25):
    params = {"interval": interval, "includeAdjustedClose": "true","events": "div,splits"}
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
                if r.status_code != 200: last_err=f"HTTP {r.status_code}"; time.sleep(0.6*(k+1)); continue
                s = _parse_chart_json(r.json())
                if isinstance(s, pd.Series) and not s.empty and len(s.dropna())>=3:
                    s.name = ticker; return s, None
                last_err = "respuesta vacía"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.6*(k+1))
    return pd.Series(dtype=float), last_err

def fetch_yahoo_range(ticker, start, end=None):
    try:
        d = yf.download(ticker, start=start, end=end, interval="1d",
                        auto_adjust=True, progress=False, threads=False)
        if d is None or d.empty: return pd.Series(dtype=float)
        s = pd.Series(d["Close"]).dropna(); s.index = pd.to_datetime(s.index).normalize()
        s.name = ticker; return s
    except Exception:
        return pd.Series(dtype=float)

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

    for col in ["Shares","Price","Fees","Taxes","FXRate","GrossAmount","NetAmount"]:
        if col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]): df[col]=np.nan
            df[col]=df[col].map(_parse_number)

    if "FXRate" in df.columns:
        df["FXRate"]=df["FXRate"].replace(0,np.nan).fillna(1.0)

    for col in ["Fees","Taxes"]:
        if col in df.columns:
            df[col]=df[col].fillna(0.0)
            df.loc[df[col].abs()>1e7, col]=0.0

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
    last_map = dict(last_hint_map or {})
    pos=[]
    for t,grp in df.groupby("Ticker"):
        sh=float(grp["SignedShares"].sum())
        if abs(sh)<1e-12: continue
        buys=grp["SignedShares"]>0
        if buys.any():
            tot_sh=float(grp.loc[buys,"SignedShares"].sum())
            cost_leg=(grp.loc[buys,"SignedShares"]*grp.loc[buys,"Price"].fillna(0)).sum()
            fees_leg=grp.loc[buys,"Fees"].fillna(0).sum()+grp.loc[buys,"Taxes"].fillna(0).sum()
            avg=(cost_leg+fees_leg)/tot_sh if tot_sh>0 else np.nan
        else:
            avg=np.nan
        px=last_map.get(t, np.nan)
        mv= sh*px if not np.isnan(px) else np.nan
        inv= sh*avg if not np.isnan(avg) else np.nan
        pl= mv-inv if not (np.isnan(mv) or np.isnan(inv)) else np.nan
        pos.append([t,sh,avg,inv,px,mv,pl])
    dfp=pd.DataFrame(pos,columns=["Ticker","Shares","AvgCost","Invested","MarketPrice","MarketValue","UnrealizedPL"])
    return dfp.sort_values("MarketValue",ascending=False)

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

# ================== SESSION MASTERS ==================
SESSION_TTL_MIN = 30  # primera carga por sesión puede sincronizar; luego solo con "Refrescar"

def need_build(name:str)->bool:
    return name not in st.session_state

def masters_expired()->bool:
    ts = st.session_state.get("_masters_built_at")
    if ts is None: return True
    return (datetime.utcnow() - ts).total_seconds() > SESSION_TTL_MIN*60

def build_masters(sync: bool):
    """Construye y guarda en session_state:
       - tx_master, settings_master
       - prices_master (wide), bench_ret_master
       - last_map_master (últimos precios)
       sync=True: intenta completar días faltantes (Yahoo) y escribir a Sheets (pesado)."""
    # Leer hojas base
    tx_df = read_sheet("Transactions")
    settings_df = read_sheet("Settings")
    benchmark = get_setting(settings_df,"Benchmark","SPY",str)

    # Tickers presentes en Transactions
    tickers = sorted(set(
        t for t in tx_df.get("Ticker", pd.Series(dtype=str)).astype(str).str.upper().str.strip().str.replace(" ","", regex=False)
        if t
    ))
    all_t = list(dict.fromkeys((tickers or []) + [benchmark]))

    # Sync pesado solo si se pidió
    if sync:
        ws, meta = _get_cache_ws()
        cache = cache_read_prices(all_t, start_date=None)
        today = pd.Timestamp(datetime.utcnow().date())
        for t in all_t:
            have = pd.Series(dtype=float)
            if cache is not None and t in cache.columns:
                have = cache[t].dropna()
            if have.empty:
                start = (today - pd.Timedelta(days=365*3)).strftime("%Y-%m-%d")
                s = fetch_yahoo_range(t, start=start, end=None)
                if s.empty:
                    s2, _ = _direct_one(t, start=start, end=None)
                    if not s2.empty: cache_append_prices(s2.to_frame())
                else:
                    cache_append_prices(s.to_frame())
            else:
                last = have.index.max()
                if pd.Timestamp(last).normalize() < today:
                    start = (pd.to_datetime(last).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    s = fetch_yahoo_range(t, start=start, end=None)
                    if s.empty:
                        s2, _ = _direct_one(t, start=start, end=None)
                        if not s2.empty: cache_append_prices(s2.to_frame())
                    else:
                        cache_append_prices(s.to_frame())

    # Construir masters desde Sheets (sin descargar otra vez de Yahoo)
    prices = cache_read_prices(all_t, start_date=None)
    bench_ret = pd.Series(dtype=float)
    if prices is not None and not prices.empty and benchmark in prices.columns:
        bench_ret = prices[[benchmark]].pct_change().dropna()[benchmark]
        prices = prices.drop(columns=[benchmark], errors="ignore")

    # Últimos precios desde cache wide
    last_map={}
    if prices is not None and not prices.empty:
        last_map = prices.ffill().iloc[-1].dropna().astype(float).to_dict()

    # Guardar en sesión
    st.session_state["tx_master"]       = tx_df
    st.session_state["settings_master"] = settings_df
    st.session_state["prices_master"]   = prices if prices is not None else pd.DataFrame()
    st.session_state["bench_ret_master"]= bench_ret
    st.session_state["last_map_master"] = last_map
    st.session_state["_masters_built_at"] = datetime.utcnow()

# ================== ELIMINAR POR TICKER ==================
def delete_transactions_by_ticker(ticker:str)->int:
    ws = open_ws("Transactions")
    values = ws.get_all_values()
    if not values: return 0
    headers = values[0]
    try:
        tcol = headers.index("Ticker")
    except ValueError:
        st.error("No encontré la columna 'Ticker' en Transactions."); return 0
    to_delete = []
    for i, row in enumerate(values[1:], start=2):
        if len(row)>tcol and str(row[tcol]).strip().upper()==ticker.strip().upper():
            to_delete.append(i)
    for ridx in reversed(to_delete): ws.delete_rows(ridx)
    return len(to_delete)

# ================== SIDEBAR ==================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio","Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Diagnóstico"])
window = st.sidebar.selectbox("Ventana histórica", ["6M","1Y","3Y","5Y","Max"], index=2)
period_map={"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5}
start_date = None if window=="Max" else (datetime.utcnow()-timedelta(days=period_map[window])).strftime("%Y-%m-%d")

# ================== PÁGINA: MI PORTAFOLIO ==================
if page=="Mi Portafolio":
    st.title("💼 Mi Portafolio")

    # Botón refrescar: fuerza sync pesado + reconstrucción de masters
    if st.button("🔄 Refrescar datos", use_container_width=False, type="secondary"):
        for k in ("tx_master","settings_master","prices_master","bench_ret_master","last_map_master","_masters_built_at"):
            st.session_state.pop(k, None)
        build_masters(sync=True)
        st.experimental_rerun()

    # Primera carga de sesión: construir masters (sync una sola vez si expiró TTL)
    if need_build("prices_master") or masters_expired():
        build_masters(sync=masters_expired())

    tx_df       = st.session_state["tx_master"]
    settings_df = st.session_state["settings_master"]
    benchmark   = get_setting(settings_df,"Benchmark","SPY",str)

    # Precios y benchmark desde masters, slicing por ventana (sin IO)
    prices_master   = st.session_state["prices_master"]
    bench_ret_full  = st.session_state["bench_ret_master"]

    if start_date and prices_master is not None and not prices_master.empty:
        prices = prices_master.loc[pd.Timestamp(start_date):]
    else:
        prices = prices_master.copy()

    # ---------- FIX: alinear benchmark por fechas sin KeyError ----------
    bench_ret = pd.Series(dtype=float)
    if bench_ret_full is not None and not bench_ret_full.empty:
        if prices is not None and not prices.empty:
            bench_ret = bench_ret_full.reindex(prices.index).ffill()  # <- cambio clave
        else:
            bench_ret = bench_ret_full
    # -------------------------------------------------------------------

    # Últimos precios
    last_hint_map = dict(st.session_state.get("last_map_master", {}))

    # Posiciones (sin tocar IO)
    pos_df = positions_from_tx(tx_df, last_hint_map=last_hint_map)
    if pos_df.empty:
        st.info("No hay posiciones válidas tras procesar Transactions."); st.stop()

    # Pesos y vistas
    total_mv = pos_df["MarketValue"].fillna(0).sum()
    w = (pos_df.set_index("Ticker")["MarketValue"]/total_mv).sort_values(ascending=False) if total_mv>0 else pd.Series(dtype=float)
    pos_df_view = pos_df.set_index("Ticker").reindex(w.index).reset_index() if len(w)>0 else pos_df.copy()

    since_buy = (pos_df.set_index("Ticker")["MarketPrice"]/pos_df.set_index("Ticker")["AvgCost"] - 1)\
                .replace([np.inf,-np.inf], np.nan)
    window_change = pd.Series(index=pos_df_view["Ticker"], dtype=float)
    if prices is not None and not prices.empty and prices.shape[0] >= 2:
        window_change = (prices.ffill().iloc[-1]/prices.ffill().iloc[0]-1)\
                        .reindex(pos_df_view["Ticker"]).fillna(np.nan)

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

    # Tabla editable sólo para borrar (evita coste de data_editor en cada rerun)
    colcfg = {"➖": st.column_config.CheckboxColumn(label="➖", help="Marcar para eliminar este ticker", width="small", default=False)}
    editor_key = f"positions_editor_{window}"
    disabled_cols = [c for c in view.columns if c!="➖"]
    edited = st.data_editor(view, hide_index=True, use_container_width=True,
                            column_config=colcfg, disabled=disabled_cols, key=editor_key)

    # Detección de click en ➖ (comparado contra estado previo)
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
                # tras borrar, reconstruir masters rápido (sin sync pesado)
                for k in ("tx_master","settings_master","prices_master","bench_ret_master","last_map_master"):
                    st.session_state.pop(k, None)
                build_masters(sync=False)
                st.experimental_rerun()
        with c2:
            if st.button("❌ No, cancelar"):
                st.session_state["delete_candidate"] = ""
                st.session_state[prev_key]["➖"] = False
                st.info("Operación cancelada.")

    # Métricas y gráficos (sólo slicing/cálculo local)
    if prices is not None and not prices.empty and prices.shape[0] >= 2 and len(w)>0:
        c_top1, c_top2 = st.columns([2,1])
        with c_top1:
            rets = prices.pct_change().dropna(how="all")
            wv = w.reindex(rets.columns).fillna(0).values if len(w)>0 else np.array([])
            port_ret = (rets*wv).sum(axis=1) if rets.shape[1] and len(wv) else pd.Series(dtype=float)

            rf = get_setting(settings_df,"RF",0.03,float)
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

# ================== OTRAS PESTAÑAS (sin cambios visuales) ==================
elif page in ["Optimizar y Rebalancear","Evaluar Candidato","Explorar / Research","Diagnóstico"]:
    st.title(page)
    st.info("Contenido no modificado. Esta sección mantiene el mismo formato de la app original.")
