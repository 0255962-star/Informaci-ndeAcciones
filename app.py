import os
import json
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Consulta de Acciones - Huizar", page_icon="📊")

# ======== Config clave desde Secrets ========
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.error('Falta GOOGLE_API_KEY en Secrets. Pega exactamente:  GOOGLE_API_KEY = "TU_CLAVE_AQUI"')
    st.stop()

# ======== Endpoints y modelos (probamos varios) ========
ENDPOINTS = [
    # Primero intentamos v1 (preferido)
    "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
    # Si falla, probamos v1beta
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
]
MODEL_CANDIDATES = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-pro",
]

def translate_to_spanish_rest(text: str) -> str:
    """Traduce al español usando REST de Gemini probando endpoints/modelos disponibles."""
    if not text:
        return ""
    # Prompt compacto para traducción
    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n" + text
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    params = {"key": API_KEY}
    headers = {"Content-Type": "application/json"}

    last_err = None
    for ep in ENDPOINTS:
        for model in MODEL_CANDIDATES:
            url = ep.format(model=model)
            try:
                r = requests.post(url, params=params, headers=headers, data=json.dumps(payload), timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    # Extraer texto (estructura estándar de generateContent)
                    candidates = data.get("candidates") or []
                    for c in candidates:
                        parts = (((c.get("content") or {}).get("parts")) or [])
                        for p in parts:
                            txt = (p.get("text") or "").strip()
                            if txt:
                                st.caption(f"Modelo OK: {model} · Endpoint: {url.split('/models/')[0].split('//')[1].split('/')[0]}")
                                return txt
                    # Si 200 pero sin texto, intenta siguiente combinación
                    last_err = f"200 sin texto con {model} en {url}"
                else:
                    # Guarda error para diagnóstico y sigue intentando
                    try:
                        last_err = f"{r.status_code} {r.json()}"
                    except Exception:
                        last_err = f"{r.status_code} {r.text}"
            except Exception as e:
                last_err = str(e)
                # Probar siguiente combinación
                continue
    raise RuntimeError(f"Error al traducir con Gemini (REST): {last_err}")

# ======== Cache yfinance ========
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

# ======== UI ========
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT").strip().upper()

# Reset traducción si cambia símbolo
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_es", None)
    st.session_state.last_symbol = stonk

info = get_info(stonk)

st.subheader("🏢 Nombre de la empresa")
st.write(info.get("longName", "N/A"))

summary = info.get("longBusinessSummary", "")
st.subheader("📝 Descripción del negocio (original / inglés)")
st.write(summary if summary else "No hay descripción disponible.")

st.write("---")
if st.button("Traducir a español 🇪🇸", use_container_width=True):
    if not summary:
        st.warning("No hay descripción para traducir.")
    else:
        with st.spinner("Traduciendo con Gemini..."):
            try:
                st.session_state.translated_es = translate_to_spanish_rest(summary)
            except Exception as e:
                st.error(f"Error al traducir con Gemini: {e}")

if st.session_state.get("translated_es"):
    st.subheader("🇪🇸 Descripción del negocio (traducción)")
    st.write(st.session_state.translated_es)
