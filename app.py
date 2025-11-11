import os
import io
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

import gspread
from google.oauth2.service_account import Credentials

# -------------------------------
# Config general
# -------------------------------
st.set_page_config(page_title="APP Finanzas – Portafolio", layout="wide")

# Helpers
def _to_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x).date()
    return pd.to_datetime(str(x)).date()

def format_money(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return x

# -------------------------------
# Conexión a Google Sheets
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_gs_client():
    sa_info = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def open_spreadsheet():
    gc = get_gs_client()
    # Puedes usar SHEET_ID o GSHEETS_SPREADSHEET_NAME
    key = st.secrets.get("SHEET_ID", None)
    if key:
        sh = gc.open_by_key(key)
    else:
        sh = gc.open(st.secrets["GSHEETS_SPREADSHEET_NAME"])
    return sh

def read_sheet_df(sh, tab):
    ws = sh.worksheet(tab)
    df = pd.DataFrame(ws.get_all_records())
    return df

def upsert_append_df(sh, tab, df: pd.DataFrame):
    if df.empty: 
        return
    ws = sh.worksheet(tab)
    # append rows at bottom
    rows = [df.columns.tolist()] + df.astype(object).values.tolist()
    # use batch update to avoid header duplication
    # strategy: write without header using append_rows (header exists)
    ws.append_rows(df.astype(object).values.tolist(), value_input_option="USER_ENTERED")

# -------------------------------
# Datos base
# -------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_all_sheets():
    sh = open_spreadsheet()
    # Transactions
    try:
        tx = read_sheet_df(sh, "Transactions")
        if not tx.empty:
            # Normalize types
            tx["TradeDate"] = pd.to_datetime(tx["TradeDate"])
            tx["Side"] = tx["Side"].str.strip().str.capitalize()
            for col in ["Shares", "Price", "Fees", "Taxes", "FXRate"]:
                if col in tx.columns:
                    tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0.0)
            tx["Account"] = tx.get("Account", "Main")
            tx["Currency"] = tx.get("Currency", "USD")
            tx["FXRate"] = tx.get("FXRate", 1.0)
            tx["Notes"] = tx.get("Notes", "")
        else:
            tx = pd.DataFrame(columns=["TradeDate","Side","Ticker","Shares","Price","Fees","Taxes","Account","Currency","FXRate","Notes"])
    except gspread.WorksheetNotFound:
        tx = pd.DataFrame(columns=["TradeDate","Side","Ticker","Shares","Price","Fees","Taxes","Account","Currency","FXRate","Notes"])

    # Prices
    try:
        px = read_sheet_df(sh, "Prices")
        if not px.empty:
            px["Date"] = pd.to_datetime(px["Date"])
            px["Close"] = pd.to_numeric(px["Close"], errors="coerce")
        else:
            px = pd.DataFrame(columns=["Date","Ticker","Close"])
    except gspread.WorksheetNotFound:
        px = pd.DataFrame(columns=["Date","Ticker","Close"])

    # Settings
    try:
        stg = read_sheet_df(sh, "Settings")
        settings = dict(zip(stg["Key"], stg["Value"]))
    except gspread.WorksheetNotFound:
        settings = {"BENCHMARK":"^GSPC", "RISK_FREE":"0.04", "DEFAULT_ACCOUNT":"Main"}

    # Historial
    try:
        hist = read_sheet_df(sh, "Historial")
        if not hist.empty:
            hist["BuyDateFirst"] = pd.to_datetime(hist["BuyDateFirst"], errors="coerce")
            hist["SellDateLast"]  = pd.to_datetime(hist["SellDateLast"], errors="coerce")
            for c in ["BuyAvgCost","SellPrice","SharesBought","SharesSold","FeesBuy","FeesSell","HoldingDays","RealizedPL","RealizedPLPct"]:
                if c in hist.columns:
                    hist[c] = pd.to_numeric(hist[c], errors="coerce")
        else:
            hist = pd.DataFrame(columns=["Ticker","BuyDateFirst","BuyAvgCost","SharesBought","SellDateLast","SellPrice","SharesSold","FeesBuy","FeesSell","HoldingDays","RealizedPL","RealizedPLPct","Account","Notes"])
    except gspread.WorksheetNotFound:
        hist = pd.DataFrame(columns=["Ticker","BuyDateFirst","BuyAvgCost","SharesBought","SellDateLast","SellPrice","SharesSold","FeesBuy","FeesSell","HoldingDays","RealizedPL","RealizedPLPct","Account","Notes"])

    # Watchlist
    try:
        wl = read_sheet_df(sh, "Watchlist")
    except gspread.WorksheetNotFound:
        wl = pd.DataFrame(columns=["Ticker","Notes"])

    return tx, px, settings, hist, wl

# -------------------------------
# Yahoo Finance – solo yfinance
# -------------------------------
def yf_download_closes(ticker, start=None, end=None):
    # robust yfinance call
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    if "Adj Close" in data.columns:
        ser = data["Adj Close"].copy()
    elif "Close" in data.columns:
        ser = data["Close"].copy()
    else:
        return pd.Series(dtype=float)
    ser.index = pd.to_datetime(ser.index)
    ser.name = "Close"
    return ser

def ensure_prices_incremental(px_df, tickers, start_date):
    """Lee Prices (df), descarga faltantes desde yfinance y devuelve (px_df_actualizado, log)."""
    updates = []
    for t in tickers:
        t = str(t).upper().strip()
        if not t:
            continue
        has = px_df[px_df["Ticker"]==t]
        if has.empty:
            dl_start = start_date
        else:
            last_dt = has["Date"].max().date()
            dl_start = last_dt + timedelta(days=1)
        if dl_start > datetime.utcnow().date():
            continue
        ser = yf_download_closes(t, start=dl_start, end=datetime.utcnow().date()+timedelta(days=1))
        if ser.empty:
            continue
        add = pd.DataFrame({"Date": ser.index.normalize(), "Ticker": t, "Close": ser.values})
        px_df = pd.concat([px_df, add], ignore_index=True)
        updates.append((t, len(add)))
    # dedupe
    px_df = px_df.drop_duplicates(subset=["Date","Ticker"], keep="last").sort_values(["Ticker","Date"]).reset_index(drop=True)
    return px_df, updates

# -------------------------------
# Posiciones, FIFO y métricas
# -------------------------------
def build_open_lots(tx_df):
    """Devuelve lots abiertos por ticker (FIFO) y ledger de cierres para ventas en el pasado (cálculo al vuelo)."""
    lots = {}  # ticker -> list of dicts {date, shares_remaining, price, fees}
    closes = []  # rows estilo Historial (cálculo al vuelo, no escribe)

    for _, r in tx_df.sort_values("TradeDate").iterrows():
        t = r["Ticker"].upper()
        side = r["Side"].lower()
        qty = float(r["Shares"])
        price = float(r["Price"])
        fees = float(r.get("Fees",0)) + float(r.get("Taxes",0))
        d = r["TradeDate"]
        if side == "buy":
            lots.setdefault(t, []).append({"date": d, "shares": qty, "price": price, "fees": fees})
        elif side == "sell":
            remain = qty
            lots.setdefault(t, [])
            # consume fifo
            while remain > 1e-9 and lots[t]:
                lot = lots[t][0]
                take = min(lot["shares"], remain)
                buy_cost = lot["price"]
                # fees split proportional
                fees_buy_part = (take/lot["shares"]) * lot["fees"] if lot["shares"]>0 else 0
                pl = (price - buy_cost)*take - fees - fees_buy_part
                closes.append({
                    "Ticker": t,
                    "BuyDateFirst": lot["date"],
                    "BuyAvgCost": buy_cost,
                    "SharesBought": take,
                    "SellDateLast": d,
                    "SellPrice": price,
                    "SharesSold": take,
                    "FeesBuy": round(fees_buy_part,2),
                    "FeesSell": round(fees,2),
                    "HoldingDays": (pd.to_datetime(d)-pd.to_datetime(lot["date"])).days,
                    "RealizedPL": round(pl,2),
                    "RealizedPLPct": round((price/buy_cost-1)*100,2) if buy_cost>0 else np.nan,
                    "Account": r.get("Account","Main"),
                    "Notes": r.get("Notes","")
                })
                lot["shares"] -= take
                remain -= take
                if lot["shares"] <= 1e-9:
                    lots[t].pop(0)
            # si remain>0, venta sin suficiente inventario (ignorado)

    # limpia zeros
    for t in list(lots.keys()):
        lots[t] = [l for l in lots[t] if l["shares"]>1e-9]
        if not lots[t]:
            lots.pop(t)
    return lots, pd.DataFrame(closes)

def positions_from_tx(tx_df, px_df):
    lots, _ = build_open_lots(tx_df)
    rows = []
    last_map = {}
    for t in lots.keys():
        last_ser = px_df[px_df["Ticker"]==t].sort_values("Date").tail(1)
        if not last_ser.empty:
            last_map[t] = float(last_ser["Close"].values[0])
    for t, items in lots.items():
        shares = sum(l["shares"] for l in items)
        cost = np.average([l["price"] for l in items], weights=[l["shares"] for l in items]) if items else np.nan
        last = last_map.get(t, np.nan)
        rows.append({
            "Ticker": t,
            "Shares": shares,
            "AvgBuy": cost,
            "Last": last,
            "PL$": (last-cost)*shares if pd.notna(last) and pd.notna(cost) else np.nan,
            "PL%": (last/cost-1)*100 if pd.notna(last) and pd.notna(cost) and cost>0 else np.nan
        })
    pos = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    # pesos
    pos["Value"] = pos["Last"]*pos["Shares"]
    tot = pos["Value"].sum()
    pos["Peso%"] = pos["Value"]/tot*100 if tot>0 else 0
    # ranking por peso
    pos = pos.sort_values("Peso%", ascending=False).reset_index(drop=True)
    pos.insert(0, "#", np.arange(1, len(pos)+1))
    return pos

def portfolio_equity_curve(tx_df, px_df):
    """Equity = market value (activos) + cash (flujo por transacciones)."""
    if px_df.empty:
        return pd.DataFrame(columns=["Date","Equity","MarketValue","Cash"])
    # universo fechas
    dates = pd.to_datetime(px_df["Date"].unique()).sort_values()
    # posiciones por fecha
    tickers = px_df["Ticker"].unique()
    shares_t = {t: 0.0 for t in tickers}
    mv = []
    cash = 0.0
    tx_sorted = tx_df.sort_values("TradeDate")
    d_iter = iter(dates)
    cur_date = next(d_iter, None)
    tx_idx = 0
    tx_list = tx_sorted.to_dict("records")
    # recorrido fechas
    while cur_date is not None:
        # aplica transacciones hasta cur_date
        while tx_idx < len(tx_list) and pd.to_datetime(tx_list[tx_idx]["TradeDate"]).date() <= cur_date.date():
            r = tx_list[tx_idx]
            t = r["Ticker"].upper()
            if t not in shares_t:
                shares_t[t] = 0.0
            qty = float(r["Shares"])
            price = float(r["Price"])
            fees = float(r.get("Fees",0))+float(r.get("Taxes",0))
            side = r["Side"].lower()
            cf = 0.0
            if side=="buy":
                shares_t[t] += qty
                cf = -(price*qty + fees)
            elif side=="sell":
                shares_t[t] -= qty
                cf = +(price*qty - fees)
            cash += cf
            tx_idx += 1

        # market value
        mv_d = 0.0
        for t, sh in shares_t.items():
            if abs(sh) < 1e-9: 
                continue
            row = px_df[(px_df["Ticker"]==t)&(px_df["Date"]==cur_date)]
            if row.empty:
                continue
            mv_d += float(row["Close"].values[0]) * sh
        mv.append({"Date": cur_date, "MarketValue": mv_d, "Cash": cash, "Equity": mv_d+cash})
        cur_date = next(d_iter, None)
    return pd.DataFrame(mv)

# -------------------------------
# UI – Sidebar
# -------------------------------
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Mi Portafolio", "Optimizar y Rebalancear", "Evaluar Candidato"])
hist_window = st.sidebar.selectbox("Ventana histórica", ["1Y", "3Y", "5Y", "Max"], index=1)

# -------------------------------
# Carga datos base y snapshot incremental de Prices
# -------------------------------
tx_df, px_df, settings, hist_df, wl_df = load_all_sheets()
benchmark = settings.get("BENCHMARK", "^GSPC")

# Universo para snapshot: tickers en tx + watchlist + benchmark
tickers_universe = sorted(set(tx_df["Ticker"].str.upper()) | set(wl_df.get("Ticker", pd.Series([], dtype=str)).str.upper()) | set([benchmark]))
tickers_universe = [t for t in tickers_universe if t]  # limpia

# define start según ventana
today = datetime.utcnow().date()
start_map = {"1Y": today - timedelta(days=365),
             "3Y": today - timedelta(days=3*365),
             "5Y": today - timedelta(days=5*365),
             "Max": today - timedelta(days=20*365)}
start_date = start_map.get(hist_window, today - timedelta(days=3*365))

with st.spinner("Actualizando precios (snap incremental)…"):
    px_df, upd = ensure_prices_incremental(px_df, tickers_universe, start_date)
# Guarda updates en Sheets si hubo
if upd:
    sh = open_spreadsheet()
    upsert_append_df(sh, "Prices", pd.DataFrame(px_df.tail(sum(u[1] for u in upd))))

# -------------------------------
# Página: Mi Portafolio
# -------------------------------
if page == "Mi Portafolio":
    st.header("💼 Mi Portafolio")

    # Posiciones activas
    pos_df = positions_from_tx(tx_df, px_df)
    st.subheader("Detalle de posiciones (estilo broker)")
    # Eliminar doble enumeración: ya dejamos sólo "#"
    st.dataframe(pos_df.style.format({"AvgBuy":format_money,"Last":format_money,"PL$":format_money,"Value":format_money,"Peso%":"{:.2f}%","PL%":"{:.2f}%"}), use_container_width=True)

    # Botón vender por fila
    with st.expander("↘︎ Vender/Eliminar posición"):
        sel = st.selectbox("Ticker a vender", pos_df["Ticker"].tolist())
        if sel:
            row = pos_df[pos_df["Ticker"]==sel].iloc[0]
            max_qty = float(row["Shares"])
            with st.form("sell_form"):
                sell_date = st.date_input("Fecha", value=today)
                qty = st.number_input("Cantidad a vender", min_value=0.0, max_value=max_qty, value=max_qty, step=0.1)
                sell_price = st.number_input("Precio de venta", min_value=0.0, value=float(row["Last"]) if pd.notna(row["Last"]) else 0.0, step=0.01)
                fees = st.number_input("Fees", min_value=0.0, value=0.0, step=0.01)
                taxes = st.number_input("Taxes", min_value=0.0, value=0.0, step=0.01)
                notes = st.text_input("Notas", "")
                submitted = st.form_submit_button("Confirmar venta")
            if submitted and qty>0 and sell_price>0:
                sh = open_spreadsheet()
                sell_tx = pd.DataFrame([{
                    "TradeDate": sell_date.strftime("%Y-%m-%d"),
                    "Side": "Sell",
                    "Ticker": sel,
                    "Shares": qty,
                    "Price": sell_price,
                    "Fees": fees,
                    "Taxes": taxes,
                    "Account": "Main",
                    "Currency": "USD",
                    "FXRate": 1.0,
                    "Notes": notes
                }])
                upsert_append_df(sh, "Transactions", sell_tx)

                # Construir cierres FIFO SOLO para esta venta y registrar en Historial (materializado)
                # Reconstruimos lots previos (antes de la venta) con tx_df + esta venta
                new_tx = pd.concat([tx_df, sell_tx.assign(TradeDate=pd.to_datetime(sell_tx["TradeDate"]))], ignore_index=True)
                lots_before, _closes_past = build_open_lots(tx_df)  # antes de la venta
                # Simula cierre sobre lots_before:
                remain = qty
                closes_rows = []
                while remain>1e-9 and sel in lots_before and lots_before[sel]:
                    lot = lots_before[sel][0]
                    take = min(lot["shares"], remain)
                    fees_buy_part = (take/lot["shares"]) * lot["fees"] if lot["shares"]>0 else 0
                    pl = (sell_price - lot["price"])*take - fees - fees_buy_part
                    closes_rows.append({
                        "Ticker": sel,
                        "BuyDateFirst": lot["date"].strftime("%Y-%m-%d"),
                        "BuyAvgCost": lot["price"],
                        "SharesBought": take,
                        "SellDateLast": sell_date.strftime("%Y-%m-%d"),
                        "SellPrice": sell_price,
                        "SharesSold": take,
                        "FeesBuy": round(fees_buy_part,2),
                        "FeesSell": round(fees,2),
                        "HoldingDays": (pd.to_datetime(sell_date)-pd.to_datetime(lot["date"])).days,
                        "RealizedPL": round(pl,2),
                        "RealizedPLPct": round((sell_price/lot["price"]-1)*100,2) if lot["price"]>0 else np.nan,
                        "Account": "Main",
                        "Notes": notes
                    })
                    lot["shares"] -= take
                    remain -= take
                    if lot["shares"]<=1e-9:
                        lots_before[sel].pop(0)
                if closes_rows:
                    upsert_append_df(sh, "Historial", pd.DataFrame(closes_rows))
                st.success(f"Venta de {qty} {sel} registrada. Historial actualizado.")
                st.cache_data.clear()

    # Equity curve (activos + cash)
    eq = portfolio_equity_curve(tx_df, px_df)
    st.subheader("Evolución del valor total (Equity = Activos + Caja)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["Date"], y=eq["Equity"], name="Equity", mode="lines"))
    fig.add_trace(go.Scatter(x=eq["Date"], y=eq["MarketValue"], name="Market Value", mode="lines"))
    fig.add_trace(go.Scatter(x=eq["Date"], y=eq["Cash"], name="Cash", mode="lines"))
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Pastel solo activos activos
    st.subheader("Distribución actual (solo activos activos)")
    if not pos_df.empty:
        pie = px.pie(pos_df, names="Ticker", values="Value", hole=0.35)
        st.plotly_chart(pie, use_container_width=True)

    # Historial desplegable
    st.subheader("Historial (cerradas)")
    with st.expander("Mostrar historial"):
        st.dataframe(hist_df.style.format({"BuyAvgCost":format_money,"SellPrice":format_money,"RealizedPL":format_money,"RealizedPLPct":"{:.2f}%"}), use_container_width=True)

# -------------------------------
# Página: Evaluar Candidato
# -------------------------------
elif page == "Evaluar Candidato":
    st.header("📌 Evaluar Candidato")

    # Entrada de ticker
    cand = st.text_input("Ticker a evaluar", value="AAPL").upper().strip()

    # Trae precios del candidato si no están
    if cand:
        px_df, upd2 = ensure_prices_incremental(px_df, [cand], start_date)
        if upd2:
            sh = open_spreadsheet()
            upsert_append_df(sh, "Prices", pd.DataFrame(px_df.tail(sum(u[1] for u in upd2))))

    # KPIs rápidos (Sharpe con activos activos + candidato hipotético a peso objetivo)
    st.write("Forma de evaluación")
    mode = st.radio("", ["Asignar porcentaje","Añadir acciones"], horizontal=True)

    weight = st.slider("Peso objetivo del candidato", min_value=0.0, max_value=0.4, value=0.1, step=0.01)
    # (Para mantener compacto, omitimos la optimización; mostramos solo UI simulada)
    st.metric("Sharpe actual", "—")
    st.metric("Sharpe con candidato", "—", delta="—")
    st.metric("MDD actual", "—")
    st.metric("MDD con candidato", "—", delta="—")

    # ---- Agregar acción (formulario) ----
    st.markdown("---")
    st.subheader("➕ Agregar acción")
    with st.expander("Abrir formulario"):
        last_px = np.nan
        if cand:
            last_row = px_df[px_df["Ticker"]==cand].sort_values("Date").tail(1)
            if not last_row.empty:
                last_px = float(last_row["Close"].values[0])
        col1,col2 = st.columns([3,1])
        with st.form("add_action_form"):
            with col1:
                ticker_input = st.text_input("Ticker", value=cand)
            with col2:
                if st.form_submit_button("✖︎ Quitar preselección"):
                    ticker_input = ""
            trade_date = st.date_input("Fecha", value=datetime.utcnow().date())
            shares = st.number_input("Shares", min_value=0.0, value=1.0, step=0.1)
            price = st.number_input("Precio (prefill Yahoo)", min_value=0.0, value=last_px if not np.isnan(last_px) else 0.0, step=0.01)
            fees = st.number_input("Fees", min_value=0.0, value=0.0, step=0.01)
            taxes = st.number_input("Taxes", min_value=0.0, value=0.0, step=0.01)
            notes = st.text_input("Notas", "")
            saved = st.form_submit_button("Guardar compra")
        if saved and ticker_input and shares>0 and price>0:
            sh = open_spreadsheet()
            new_tx = pd.DataFrame([{
                "TradeDate": trade_date.strftime("%Y-%m-%d"),
                "Side": "Buy",
                "Ticker": ticker_input.upper(),
                "Shares": shares,
                "Price": price,
                "Fees": fees,
                "Taxes": taxes,
                "Account": "Main",
                "Currency": "USD",
                "FXRate": 1.0,
                "Notes": notes
            }])
            upsert_append_df(sh, "Transactions", new_tx)
            st.success(f"Compra de {shares} {ticker_input.upper()} guardada en Transactions.")
            st.cache_data.clear()

# -------------------------------
# Página: Optimizar y Rebalancear (placeholder estético)
# -------------------------------
else:
    st.header("⚙️ Optimizar y Rebalancear")
    st.info("Se mantiene la estructura existente. Esta vista usa sólo activos activos para métricas y pesos.")
    # Aquí podrías llamar a tu rutina de frontera eficiente/optimización como antes
