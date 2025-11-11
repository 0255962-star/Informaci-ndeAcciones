# APP Finanzas – Portafolio

### Secretos en Streamlit
Configura en `secrets`:

GSHEETS_SPREADSHEET_NAME = "<NOMBRE O ID DE TU SHEET>"

[gcp_service_account]
type = "service_account"
project_id = "<...>"
private_key_id = "<...>"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "<...>@<project>.iam.gserviceaccount.com"
client_id = "<...>"
token_uri = "https://oauth2.googleapis.com/token"

### Primer uso
1. Crea la hoja con las pestañas `Transactions, Historial, Prices, Settings, Watchlist` (o usa la plantilla .xlsx y conviértela a Sheets).
2. Comparte el Sheet con el `client_email` del service account (Editor).
3. Despliega en Streamlit Cloud y listo.

### Notas clave
- `Prices` es histórico incremental (append-only).
- La app lee primero de `Prices` y solo llama a Yahoo si faltan días.
- “Evaluar Candidato” permite **Agregar acción** (BUY).
- En “Mi Portafolio” puedes **Vender** (parcial o total). La venta va a `Transactions` y el cierre FIFO se **materializa** en `Historial`.
