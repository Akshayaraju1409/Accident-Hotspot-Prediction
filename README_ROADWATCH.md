# RoadWatch — React road-safety intelligence dashboard

This redesign adds a React frontend and FastAPI adapter while retaining the original Python prediction pipeline, model artifacts, data preparation, hotspot detection, SHAP explainer and Open-Meteo integration. The original Streamlit app remains available as `app.py`.

## Architecture

- `frontend/`: React + Vite + React-Leaflet dashboard; dark CARTO basemap with OpenStreetMap data attribution.
- `api/main.py`: FastAPI endpoints that call the existing `predict.Service` directly.
- `predict.py`, `features.py`, `data_prep.py`, `hotspots.py`, `modeling.py`, `train.py`: original prediction and training pipeline, unchanged.
- `artifacts/`: existing trained models and context. No retraining is required if these artifacts are present.

## Run locally

Terminal 1 — Python API (from the `accident_hotspot_v2` project directory):

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — React UI:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The API health check is http://localhost:8000/api/health.

If the model artifacts are missing, run `python train.py` from the project root first. If you want to use the existing Streamlit version, run `streamlit run app.py`.

## Map tiles and API keys

The new UI uses React-Leaflet to render CARTO's dark basemap tiles, with OpenStreetMap attribution. It does not require a Google Maps API key. This is a hosted tile service, so check the provider's current usage terms before production deployment. For higher-scale deployment, use a dedicated tile provider such as MapTiler or self-hosted tiles and configure the `TileLayer` URL in `frontend/src/main.jsx`.

## API

- `GET /api/health` — verifies that trained artifacts load and lists supported cities.
- `GET /api/overview` — returns dataset/model diagnostics.
- `POST /api/forecast` — calls the original `Service.forecast()` method. Supports scenario weather or live Open-Meteo weather.
- `POST /api/explain` — calls the original SHAP-based `Service.explain()` method for a selected cell.

Example forecast body:

```json
{
  "city": "Bangalore",
  "date": "2026-11-07",
  "hour": 12,
  "horizon": 6,
  "weather_mode": "scenario",
  "weather": "rain",
  "density": "medium"
}
```

## Data validity note

The existing README and diagnostics document that the bundled `indian_roads_dataset.csv` appears synthetic and lacks meaningful spatial/temporal crash clustering. The UI therefore presents outputs as model estimates and includes no claim that the mapped cells are verified real-world blackspots. Validate with genuine geocoded crash records before using for operational road-safety decisions.


## RoadWatch V4 India data upgrade
See `V4_README.md` for the separate Indian highway and Bengaluru data pipeline, commands, endpoints and research limitations. The original V3 React interface and prediction pipeline remain preserved.
