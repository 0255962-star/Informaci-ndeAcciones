# 📊 Consulta de Acciones – MODELO FINANCIERO GASCON

Mini app en **Streamlit** que:
- Consulta datos de una acción usando **YFinance**.
- Traduce la descripción con **Gemini (Google AI Studio)**.
- Muestra una **gráfica de velas (candlestick)** interactiva con **Plotly** y volumen.
- Selector de periodos: **1 semana, 1 mes, 6 meses, YTD, 1 año, 3 años, 5 años**.

---

## 🚀 Instalación local

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

