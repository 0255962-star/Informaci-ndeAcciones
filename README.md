# APP Finanzas – Portafolio

App de Streamlit conectada a Google Sheets (como base de datos) y a Yahoo Finance (vía `yfinance`) para:
- Ver posiciones del portafolio
- Optimizar (frontera eficiente, Sharpe)
- Evaluar un candidato sin modificar el portafolio
- Explorar un ticker puntual

## Secrets requeridos (Streamlit Cloud)
```toml
# .streamlit/secrets.toml (ejemplo)
GSHEETS_SPREADSHEET_NAME="TuSpreadsheetID_o_Nombre"
GOOGLE_API_KEY="(si lo usas en otra parte)"

[gcp_service_account]
type="service_account"
project_id="xxx"
private_key_id="xxx"
private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email="xxx@xxx.iam.gserviceaccount.com"
client_id="..."
token_uri="https://oauth2.googleapis.com/token"
