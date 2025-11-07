# Finanzas – Portafolio Activo (Streamlit)

App de portafolios orientada a estudiantes de finanzas:
- **Home (Mi Portafolio)**: KPIs, asignación, crecimiento y sugerencias.
- **Optimizar y Rebalancear**: Markowitz + tabla de “órdenes sugeridas”.
- **Evaluar Candidato**: impacto (ΔSharpe, Δvol, ΔMDD, correlaciones) con explicación.
- **Explorar / Research**: velas + SMAs y resumen traducido con Gemini.
- **Herramientas**: β/α, CAPM, Monte Carlo.

## 1) Secrets
En **Streamlit Cloud** o local, define en `st.secrets`:

```toml
# .streamlit/secrets.toml (o en el panel de Secrets de Streamlit Cloud)
SHEET_ID = "TU_SHEET_ID"         # ID del Google Sheets
GEMINI_API_KEY = "opcional"      # Para traducción / resumen
[gcp_service_account]
type = "service_account"
project_id = "XXXX"
private_key_id = "XXXX"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "xxxx@xxxx.iam.gserviceaccount.com"
client_id = "XXXX"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/xxxx.iam.gserviceaccount.com"

