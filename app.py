import os
import streamlit as st
import yfinance as yf

# ====== Config Gemini ======
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Modelo principal y alternativas por si tu cuenta/región no tiene el primero
PRIMARY_MODEL = "gemini-1.5-flash"
FALLBACK_MODELS = ["gemini-1.5-flash-8b", "gemini-1.5-pro"]

def translate_with_gemini_to_spanish(text: str) -> str:
    """
    Traduce 'text' al ESPAÑOL con Gemini. Prueba modelos alternos si el principal falla.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GOOGLE_API_KEY en Secrets.")

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n"
        f"{text}"
    )

    last_err = None
    for model_id in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            model = genai.GenerativeModel(model_id)
            resp = model.generate_content(prompt)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            last_err = e
            # intenta con el siguiente modelo
            continue
    raise RuntimeError(f"Error al traducir con Gemini: {last_err}")

# ====== UI ======
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT")

# Si cambia el símbolo, limpiamos la traducción guardada
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
                st.session_state.translated_es = translate_with_gemini_to_spanish(summary)
            except Exception as e:
                st.error(str(e))

if st.session_state.get("translated_es"):
    st.subheader("🇪🇸 Descripción del negocio (traducción)")
    st.write(st.session_state.translated_es)

