# app.py — Diagnóstico de conexiones (Gemini + Google Sheets)
import re
import json
import requests
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Diagnóstico: Gemini & Google Sheets", layout="centered")
st.title("Diagnóstico de Conexiones")
st.caption("Verifica que tus secrets de Gemini y Google Sheets estén bien configurados.")

# -------------------------------------------------------------------
# 0) Leer secrets (sin mostrar valores)
# -------------------------------------------------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
SHEET_TARGET = (
    st.secrets.get("GSHEETS_SPREADSHEET_NAME", "")     # nombre o URL
    or st.secrets.get("GSHEETS_SPREADSHEET_ID", "")     # opcional: ID puro
)
WORKSHEET_NAME = st.secrets.get("GSHEETS_WORKSHEET_NAME", "Portafolio")

st.subheader("Resumen de secrets encontrados")
c1, c2, c3, c4 = st.columns(4)
c1.write(f"**GOOGLE_API_KEY:** {'OK' if GOOGLE_API_KEY else 'FALTA'}")
c2.write(f"**GSHEETS_SPREADSHEET_NAME/ID:** {'OK' if SHEET_TARGET else 'FALTA'}")
c3.write(f"**WORKSHEET:** {WORKSHEET_NAME or '(vacío)'}")
c4.write(f"**gcp_service_account:** {'OK' if 'gcp_service_account' in st.secrets else 'FALTA'}")

st.info(
    "Si algo marca **FALTA**, corrígelo en `.streamlit/secrets.toml` o en *Settings → Secrets* de Streamlit Cloud."
)

st.markdown("---")

# -------------------------------------------------------------------
# 1) Verificación de GEMINI (Google Generative Language API)
# -------------------------------------------------------------------
st.header("Gemini (Google Generative Language API)")

if not GOOGLE_API_KEY:
    st.error("Falta `GOOGLE_API_KEY` en secrets.")
else:
    BASES = [
        "https://generativelanguage.googleapis.com/v1",
        "https://generativelanguage.googleapis.com/v1beta",
    ]
    PREFERRED = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    def list_models(base: str):
        url = f"{base}/models"
        try:
            r = requests.get(url, params={"key": GOOGLE_API_KEY}, timeout=20)
            if r.status_code == 200:
                return True, r.json()
            return False, r.text
        except Exception as e:
            return False, str(e)

    def pick_model_and_base():
        last_err = None
        for base in BASES:
            ok, data = list_models(base)
            if not ok:
                last_err = data
                continue
            models = data.get("models", []) if isinstance(data, dict) else []
            supported = set()
            for m in models:
                name = m.get("name")
                methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
                if not name:
                    continue
                short = name.split("/")[-1]
                if any(x in ("generateContent", "generate_content") for x in methods):
                    supported.add(short)
            for pref in PREFERRED:
                if pref in supported:
                    return base, pref
            if supported:
                return base, sorted(supported)[0]
            last_err = "No hay modelos con generateContent en este endpoint."
        raise RuntimeError(f"No se pudo seleccionar modelo/base. Último error: {last_err}")

    def generate_content_rest(base: str, model: str, text: str) -> str:
        url = f"{base}/models/{model}:generateContent"
        payload = {"contents": [{"parts": [{"text": text}]}]}
        headers = {"Content-Type": "application/json"}
        params = {"key": GOOGLE_API_KEY}
        r = requests.post(url, params=params, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise RuntimeError(f"{r.status_code} {msg}")
        data = r.json()
        for cand in data.get("candidates", []):
            content = cand.get("content") or {}
            parts = content.get("parts") or []
            for p in parts:
                t = (p.get("text") or "").strip()
                if t:
                    return t
        return ""

    if st.button("Probar conexión a Gemini"):
        try:
            base, model = pick_model_and_base()
            st.success(f"Conexión a Gemini OK · Endpoint: {base} · Modelo: {model}")

            prueba = "Traduce al español: 'The quick brown fox jumps over the lazy dog.'"
            out = generate_content_rest(base, model, prueba)
            st.write("**Respuesta de prueba:**")
            st.code(out or "(vacío)")
            if "zorro" in (out or "").lower():
                st.success("Traducción de prueba correcta ✅")
            else:
                st.warning("La llamada respondió, pero el texto no parece traducido. Revisa modelo o clave.")
        except Exception as e:
            st.error(f"Fallo en la conexión a Gemini: {e}")

st.markdown("---")

# -------------------------------------------------------------------
# 2) Verificación de GOOGLE SHEETS (Service Account)
# -------------------------------------------------------------------
st.header("Google Sheets")

if "gcp_service_account" not in st.secrets:
    st.error("Falta el bloque `[gcp_service_account]` en secrets.")
else:
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    def parse_spreadsheet_target(text: str):
        if not text:
            return None, None
        # URL completa → ID
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
        if m:
            return None, m.group(1)
        # ID “puro”
        if re.fullmatch(r"[A-Za-z0-9\-_]{20,}", text):
            return None, text
        # de lo contrario, es el nombre
        return text, None

    with st.form("sheets_form"):
        st.caption("Puedes sobreescribir lo de secrets aquí para la prueba.")
        input_target = st.text_input("Spreadsheet (nombre, URL o ID)", value=SHEET_TARGET)
        input_ws = st.text_input("Worksheet", value=WORKSHEET_NAME)
        submitted = st.form_submit_button("Probar conexión a Sheets")

    if submitted:
        try:
            creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
            client = gspread.authorize(creds)
            st.success("Autenticación con Service Account OK ✅")
        except Exception as e:
            st.error(f"No se pudo autenticar con el Service Account: {e}")
            st.stop()

        title, key = parse_spreadsheet_target(input_target)
        try:
            sh = client.open_by_key(key) if key else client.open(title)
            ws = sh.worksheet(input_ws)
            data = ws.get_all_records()
            df = pd.DataFrame(data)
            st.success(f"Acceso a hoja OK ✅ · Spreadsheet: {sh.title} · Worksheet: {input_ws}")
            if df.empty:
                st.warning("La hoja existe, pero no tiene filas. Agrega datos (ej. columnas Ticker, Weight).")
            else:
                st.write("Primeras filas:")
                st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo abrir/leer el Spreadsheet/Worksheet: {e}")
            st.info("Checklist: 1) Comparte el Sheet con el client_email del service account. "
                    "2) Verifica nombre/URL/ID y nombre de worksheet. "
                    "3) Asegura columnas esperadas (p.ej., Ticker, Weight).")

st.markdown("---")
st.caption("Tip: cuando todo esté OK, vuelve a poner tu app original.")
