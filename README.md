# 📊 Consulta de Acciones – MODELO FINANCIERO GASCON

Mini app en **Streamlit** que:
- Consulta datos de una acción con **yfinance**.
- Traduce la descripción del negocio con **Gemini** (Google AI Studio).
- Muestra **gráfica de velas (candlestick)** con **OHLC** y **Volumen**.
- Selector de periodos: **1 semana, 1 mes, 6 meses, YTD, 1 año, 3 años, 5 años**.

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
