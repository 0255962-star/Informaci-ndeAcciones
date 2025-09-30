import os
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# =========================
#  Configuración / Diagnóstico
# =========================
st.set_page_config(page_title="Consulta de Acciones - Huizar", page_icon="📊", layout="centered")

# Muestra la versión del SDK de Gemini para verificar que tomó el requirements correcto
st.caption(f"google-generativeai version: {getattr(genai, '__version__', 'unknown')}")

# Lee la API key desde Streamlit Secrets o variable de entorno
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GEMINI_API_KEY:
    st.error("Falta GOOGLE_API_KEY en Secrets. Ve a Settings → Secrets y pega:  GOOGLE_API_KEY = \"TU_CLAVE_AQUI\"")
    st.stop()

# Configura el SDK
genai.configure(api_key=GEMINI_API_KEY)

# Usa un modelo seguro para v1beta: texto puro
MODEL_ID = "gemini-pro"
st.caption(f"Usando modelo: {MODEL_ID}")

# =========================
#  Funciones
# =========================
def translate_to_spanish(text: str) -> str:
    """
    Traduce 'text' al ESPAÑOL con Gemini (modelo gemini-pro).
    """
    if not text:
        return ""
    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n"
        f"{text}"
    )
    model = genai.GenerativeModel(MODEL_ID)
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()


@st.cache_data(ttl=3600)
def get_info(symbol: str):
    """
    Obtiene info del ticker desde yfinance. Cacheada por 1 hora.
    """
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


# =========================
#  UI
# =========================
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

# Entrada del ticker
stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT").strip().upper()

# Control de estado para limpiar traducción cuando cambie el ticker
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_es", None)
    st.session_state.last_symbol = stonk

# Carga de información
info = get_info(stonk)

# Nombre de la empresa
st.subheader("🏢 Nombre de la empresa")
st.write(info.get("longName", "N/A"))

# Descripción original (normalmente en inglés en Yahoo Finance)
summary = info.get("longBusinessSummary", "")
st.subheader("📝 Descripción del negocio (original / inglés)")
st.write(summary if summary else "No hay descripción disponible.")

st.write("---")
# Botón para traducir al español
if st.button("Traducir a español 🇪🇸", use_container_width=True):
    if not summary:
        st.warning("No hay descripción para traducir.")
    else:
        with st.spinner("Traduciendo con Gemini..."):
            try:
                st.session_state.translated_es = translate_to_spanish(summary)
            except Exception as e:
                st.error(f"Error al traducir con Gemini: {e}")

# Mostrar traducción si existe
if st.session_state.get("translated_es"):
    st.subheader("🇪🇸 Descripción del negocio (traducción)")
    st.write(st.session_state.translated_es)

# (Opcional) Diagnóstico de modelos disponibles: descomenta si necesitas depurar
# try:
#     names = [m.name for m in genai.list_models()]
#     with st.expander("Modelos disponibles (debug)"):
#         st.write(names[:100])
# except Exception as e:
#     st.caption(f"No se pudo listar modelos: {e}")
