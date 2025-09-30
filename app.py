import os
import streamlit as st
import yfinance as yf

# ====== Config Gemini ======
# Lee la API key desde st.secrets o variable de entorno
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

def translate_with_gemini(text: str) -> str:
    """
    Traduce 'text' al inglés con Gemini. Requiere GOOGLE_API_KEY.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Configúrala en st.secrets o como variable de entorno."
        )
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Translate the following Spanish business description to natural, "
            "concise business English. Keep company/product names and numbers. "
            "Only return the translation, no preface or explanation.\n\n"
            f"{text}"
        )
        resp = model.generate_content(prompt)
        # .text puede ser None si no hay salida válida
        return (resp.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"Error al traducir con Gemini: {e}")

# ====== UI ======
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

# Entrada con valor por defecto
stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT")

# Reset de traducción si cambia el símbolo
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_en", None)
    st.session_state.last_symbol = stonk

# Cache para info de la empresa
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

info = get_info(stonk)

# Nombre
st.subheader("🏢 Nombre de la empresa")
st.write(info.get("longName", "N/A"))

# Descripción (original)
summary = info.get("longBusinessSummary", "")
st.subheader("📝 Descripción del negocio (original)")
st.write(summary if summary else "No hay descripción disponible.")

# Botón: Traducir a inglés con Gemini
st.write("---")
cols = st.columns([1, 2])
with cols[0]:
    translate_clicked = st.button("Traducir a inglés 🇬🇧", use_container_width=True)

if translate_clicked:
    if not summary:
        st.warning("No hay descripción para traducir.")
    else:
        with st.spinner("Traduciendo con Gemini..."):
            try:
                st.session_state.translated_en = translate_with_gemini(summary)
            except Exception as e:
                st.error(str(e))

# Muestra traducción si existe
if st.session_state.get("translated_en"):
    st.subheader("🇬🇧 Business Description (English)")
    st.write(st.session_state.translated_en)
