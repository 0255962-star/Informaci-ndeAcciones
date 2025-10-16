import os
import re
import json
import streamlit as st

# ====== Google ======
from google.oauth2.service_account import Credentials
import gspread
import requests

st.set_page_config(page_title="Prueba de conexión", page_icon="✅", layout="centered")
st.title("Prueba de conexión")

st.write("Esta página solo verifica que los *secrets* estén bien y que se puede acceder a Google Sheets (pestaña **Transactions**) y a Gemini (Google AI).")

st.markdown("---")

# =========================
# 1) PRUEBA: Google Sheets
# =========================
st.header("1) Google Sheets")

OK = "✅"
FAIL = "❌"

def parse_spreadsheet_target(text: str):
    if not text:
        return None, None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if m:
        return None, m.group(1)  # URL -> ID
    if re.fullmatch(r"[A-Za-z0-9\-_]{20,}", text):
        return None, text         # ID puro
    return text, None            # título

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 1.1 Cargar secrets
gs_target = st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")
ws_name   = st.secrets.get("GSHEETS_TRANSACTIONS_WS", "Transactions")
sa_info   = st.secrets.get("gcp_service_account", None)

c1, c2 = st.columns(2)
with c1: st.write("Spreadsheet (secrets):", f"`{gs_target}`")
with c2: st.write("Worksheet (secrets):", f"`{ws_name}`")

if not gs_target:
    st.error(f"{FAIL} Falta `GSHEETS_SPREADSHEET_NAME` en secrets.")
else:
    st.success(f"{OK} Encontré `GSHEETS_SPREADSHEET_NAME`.")

if not sa_info:
    st.error(f"{FAIL} Falta bloque `[gcp_service_account]` en secrets.")
else:
    st.success(f"{OK} Encontré `gcp_service_account`.")

# 1.2 Autenticación
client = None
if sa_info:
    try:
        creds = Credentials.from_service_account_info(dict(sa_info), scopes=SCOPES)
        client = gspread.authorize(creds)
        st.success(f"{OK} Autenticación con Service Account correcta.")
        st.caption(f"Service account: {sa_info.get('client_email','(sin email)')}")
    except Exception as e:
        st.exception(e)

# 1.3 Abrir el Spreadsheet
sh = None
if client and gs_target:
    try:
        title, key = parse_spreadsheet_target(gs_target)
        if key:
            sh = client.open_by_key(key)
            st.success(f"{OK} Abrí el spreadsheet por **ID/URL**.")
        else:
            sh = client.open(title)
            st.success(f"{OK} Abrí el spreadsheet por **título**.")
        st.write("Pestañas encontradas:", [w.title for w in sh.worksheets()])
    except Exception as e:
        st.error(f"{FAIL} No pude abrir el spreadsheet. Revisa ID/URL, permisos o si compartiste el archivo con la service account.")
        st.exception(e)

# 1.4 Abrir worksheet "Transactions" y validar encabezados
ws = None
if sh:
    try:
        ws = sh.worksheet(ws_name)
        st.success(f"{OK} Abrí la pestaña `{ws_name}`.")
        values = ws.get_all_values()
        if not values:
            st.warning("La pestaña está vacía. Debe tener al menos la fila de encabezados.")
        else:
            header = [c.strip() for c in values[0]]
            st.write("Encabezados detectados:", header)
            needed = {"Ticker","TradeDate","Shares","Price","Fees","Notes"}
            if needed.issubset(set(header)):
                st.success(f"{OK} Encabezados correctos.")
            else:
                st.warning("Encabezados incompletos o distintos. Usa exactamente: `Ticker | TradeDate | Shares | Price | Fees | Notes`")
            st.write("Primeras filas:", values[:5])
    except Exception as e:
        st.error(f"{FAIL} No pude abrir la pestaña `{ws_name}`.")
        st.exception(e)

# 1.5 (Opcional) Probar escritura controlada
if ws:
    st.markdown("**Prueba opcional de escritura** (agrega y elimina una fila de test)")
    do_write = st.toggle("Habilitar prueba de escritura", value=False)
    if do_write:
        if st.button("Ejecutar prueba de escritura"):
            try:
                test_row = ["TEST","2020-01-01","1","","","ping from app"]
                ws.append_row(test_row, value_input_option="USER_ENTERED")
                st.success(f"{OK} Fila de prueba agregada.")
                st.info("Te recomiendo borrar manualmente esa fila en el Sheet.")
            except Exception as e:
                st.error(f"{FAIL} No pude escribir en la hoja (append_row). Verifica permisos.")
                st.exception(e)

st.markdown("---")

# =========================
# 2) PRUEBA: Gemini (Google AI)
# =========================
st.header("2) Gemini (Google AI)")
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.warning("No encuentro `GOOGLE_API_KEY` en secrets. Solo se puede probar Sheets.")
else:
    st.success(f"{OK} Encontré GOOGLE_API_KEY.")
    BASES = [
        "https://generativelanguage.googleapis.com/v1",
        "https://generativelanguage.googleapis.com/v1beta",
    ]
    try:
        # Listar modelos en el primer endpoint disponible
        listed = False
        for base in BASES:
            r = requests.get(f"{base}/models", params={"key": API_KEY}, timeout=20)
            if r.status_code == 200:
                data = r.json()
                names = [m.get("name","").split("/")[-1] for m in data.get("models",[])]
                st.success(f"{OK} Conectado a {base}. Modelos detectados (primeros 10): {names[:10]}")
                listed = True
                break
        if not listed:
            st.warning("La API respondió pero no pude listar modelos. Verifica que tu API key tenga acceso a Google AI Studio.")
    except Exception as e:
        st.error(f"{FAIL} No pude llamar a la API de modelos Gemini.")
        st.exception(e)

st.markdown("---")
st.caption("Si todos los checks de arriba salen en verde, tu conexión está bien. Si algo falla, corrige ese paso y vuelve a ejecutar.")
