# Severity upgrade: ablation, model comparison, error analysis, API + UI

Unzip this folder over your project root (the folder that contains `api/`, `v4_pipeline/`, `frontend/`, `tests/`). It overwrites
`v4_pipeline/severity_v2.py`, `api/main.py`, `frontend/src/main.jsx` and `frontend/src/style.css`, and adds the other files.

| File | Status |
|---|---|
| `v4_pipeline/severity_v2.py` | replaced (date-parsing fix, row guard, bootstrap CI, pandas warning fixed) |
| `v4_pipeline/severity_analysis.py` | NEW: feature ablation, model comparison, false-negative analysis, report |
| `api/severity.py` | NEW: FastAPI router `/api/v4/severity/*` |
| `api/main.py` | replaced (mounts the router; `/api/v4/highway/metrics` now serves the V2 metrics) |
| `frontend/src/Severity.jsx` | NEW: "Crash severity" page |
| `frontend/src/main.jsx`, `style.css` | replaced (nav item, page hook, styles) |
| `tests/test_severity_*.py` | NEW: 13 tests |

## Run order (from the project root)
```bash
pip install fastapi "uvicorn[standard]" httpx matplotlib scipy      # httpx is only for the API tests
python -m v4_pipeline.severity_v2          # retrain + save the model
python -m v4_pipeline.severity_analysis    # writes artifacts_v4/analysis/ (report, tables, figures)
python -m pytest -q tests
uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev                 # open http://localhost:5173 -> "Crash severity"
```

## Outputs of the analysis (`artifacts_v4/analysis/`)
- `severity_analysis_report.md`: paste-ready tables and plain-language findings for your report
- `fig_ablation.png`, `fig_errors.png`: figures
- `false_negatives.csv`: every severe test crash the model missed, with decoded labels
- `severity_analysis.json`: same content, served by `GET /api/v4/severity/analysis` and shown in the UI

## API
- `GET  /api/v4/severity/schema`: form fields, allowed codes with labels, model info, limitations
- `POST /api/v4/severity/predict`: body `{"date":"2023-03-18","time":"23:30","chainage_km":310,"vehicle_1":6,"weather":2}`;
  all other fields optional (missing values are filled with the most typical training value)
- `GET  /api/v4/severity/metrics`, `GET /api/v4/severity/analysis`
