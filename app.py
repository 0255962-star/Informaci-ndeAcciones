import os
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# Muestra la versión del SDK para verificar que NO es v1beta (debería verse 0.8.x)
st.caption(f"google-generativeai version: {getattr(genai, '__version__', 'unknown')}")

GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GEMINI_API_KEY:
    st.stop()  # evita seguir si no hay clave

genai.configure(api_key=GEMINI_API_KEY)

# Lista de preferencia (incluye sufijo -latest y alternativas)
PREFERRED_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b-latest",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.0-pro",     # compatibilidad con claves antiguas
    "gemini-pro"          # fallback extremo (SDKs viejos)
]

def pick_supported_model() -> str:
    # Filtra modelos que soporten generateContent
    try:
        models = genai.list_models()
        supported = {m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])}
        for m in PREFERRED_MODELS:
            # Los nombres retornan como "models/xxx"
            if f"models/{m}" in supported:
                return m
    except Exception:
        pass
    # Si no se puede listar (p. ej. permisos), intenta por orden
    return PREFERRED_MODELS[0]

MODEL_ID = pick_supported_model()
st.caption(f"Usando modelo: {MODEL_ID}")

def translate_to_spanish(text: str) -> str:
    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n"
        f"{text}"
    )
    model = genai.GenerativeModel(MODEL_ID)
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()

# --------- UI base (lo que ya tenías) ----------
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")
stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT")

if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_es", None)
    st.session_state.last_symbol = stonk

@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

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
                st.session_state.translated_es = translate_to_spanish(summary)
            except Exception as e:
                st.error(f"Error al traducir con Gemini: {e}")

if st.session_state.get("translated_es"):
    st.subheader("🇪🇸 Descripción del negocio (traducción)")
    st.write(st.session_state.translated_es)
