# 📊 Consulta de Acciones – MODELO FINANCIERO GASCON

Mini app en **Streamlit** que consulta datos de una acción con **yfinance** y traduce la descripción al inglés usando **Gemini**.

## Requisitos
- Python 3.10+ recomendado
- Clave de API de Gemini (Google AI Studio)

## Instalación local (macOS/Linux/Windows)
```bash
# clona el repo
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

# crea y activa entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate

# instala dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
