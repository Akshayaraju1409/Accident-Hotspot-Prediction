# RoadWatch — Accident Hotspot Analysis and Severity Prediction (India)

RoadWatch is an academic road-safety analytics project for Indian road-accident data. It contains:

1. **A spatio-temporal hotspot pipeline** (grid cells, Gi\* and DBSCAN hotspot detection, a crash-risk model, SHAP explanations) with a FastAPI backend and a React dashboard.
2. **A planted-hotspot benchmark** that checks whether the pipeline can find hotspots when they really exist.
3. **A V4 India data pipeline** with an accident **severity classifier** on real Indian highway records and an analysis of Bengaluru Traffic Police station-level data.

> **Honest status.** The bundled city dataset (`data/indian_roads_dataset.csv`) appears synthetic and has no real spatial clustering, so the city risk model learns little beyond each city's base crash rate (see [Results](#results)). The hotspot pipeline itself is validated on the planted benchmark. The real-data severity model is a weak research baseline. None of this is suitable for operational road-safety decisions.

---

## Quick start

Requires Python 3.10+ and Node.js 18+.

### 1. Backend (FastAPI)

```bash
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Check it is running: <http://127.0.0.1:8000/api/health> should return `{"status":"ok", ...}`.

The API loads `artifacts/models.joblib` and `artifacts/context.joblib`, which are included in the repository. If they are missing or fail to load (for example after a major library upgrade), regenerate them:

```bash
python train.py
```

### 2. Frontend (React + Vite)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend calls the API at `http://localhost:8000/api`.

### 3. Optional Streamlit app

```bash
streamlit run app.py
```

### 4. Tests

```bash
python -m pytest -q tests
```

---

## Results

### City hotspot / crash-risk model (synthetic city dataset)

| Check | Result |
|---|---|
| Test PR-AUC lift, final model | 1.65 |
| Test PR-AUC lift, city-only base-rate baseline | 1.64 |
| Within-city lift vs shuffled-score null | about 1.0 (not above the null 95th percentile) |
| Leave-one-city-out lift | 0.98 to 1.05 |

The model's apparent lift comes from per-city base rates, not from locating hotspots. This is a limitation of the data, not of the pipeline (see the benchmark below). `diagnostics.py` produces these checks.

### Planted-hotspot benchmark (ground truth known)

`make_benchmark.py` adds known hotspots to the real rows and `validate_benchmark.py` measures recovery:

| Metric | Result |
|---|---|
| DBSCAN recall of planted hotspot cells | 1.00 (59 of 59 cells) |
| DBSCAN precision within 1 cell | 1.00 |
| Gi\* per-hotspot recall | 0.66 |
| Crashes captured in model's top 10% of cells | 23.3% (random = 10%, oracle ceiling = 24.8%) |

```bash
python make_benchmark.py
python validate_benchmark.py     # writes artifacts_benchmark/validation.json
```

To run the whole pipeline on the benchmark file instead of the default data:

```powershell
$env:HOTSPOT_CSV="data/benchmark_planted.csv"; $env:HOTSPOT_ART="artifacts_benchmark"; python train.py
```

### Severity V2 (real Indian highway records)

Binary task: **Severe** = Fatal or Grievous injury, **Non-severe** = Minor or No injury. Chronological train/validation/test split; model and threshold chosen on validation data only.

| Metric (held-out test, 362 records) | Result |
|---|---:|
| Selected model / threshold | Extra Trees / 0.43 |
| Accuracy | 57.46% |
| Balanced accuracy | 58.43% |
| Severe precision | 49.75% |
| Severe recall | 64.71% (99 of 153 severe cases) |
| Severe F2 | 61.04% |
| Non-severe specificity | 52.15% |

A four-class Extra Trees reference reaches 30.11% accuracy and 24.56% macro F1. `Causes_D` is excluded because it may contain post-accident information (target leakage). Probabilities are uncalibrated.

---

## V4 India data pipeline

Run from the project root:

```bash
python -m v4_pipeline.download_data        # downloads Zenodo highway CSV + Bengaluru data (needs internet)
python -m v4_pipeline.prepare_highway
python -m v4_pipeline.prepare_bengaluru
python -m v4_pipeline.highway_hotspots     # writes a status file: chainage hotspots cannot be mapped validly
python -m v4_pipeline.train_severity       # baseline severity model
python -m v4_pipeline.severity_v2          # final binary severity experiment (Severity V2)
```

Optional extra algorithm for the benchmark script: `pip install -r requirements-v4-optional.txt` (CatBoost).

Outputs go to `data/v4/processed/` (cleaned CSVs are git-ignored and regenerated) and `artifacts_v4/` (metrics JSON/CSV tracked; model binaries are git-ignored).

**Data sources**

- Highway accidents: Zenodo record 16946653 (DOI 10.5281/zenodo.16946653), 13 columns, about 8,100 raw rows, about 4,050 after removing exact duplicates.
- Associated study (partial codebook for coded fields): <https://pmc.ncbi.nlm.nih.gov/articles/PMC12686657/>
- Bengaluru road crashes (Bengaluru Traffic Police via OpenCity): <https://data.opencity.in/dataset/bengaluru-road-crashes-data>. Station-level aggregates, not GPS crash points.

**Source interpretation notes**

- `Accident_Location_A` is an urban/rural classification, not a corridor ID. The row-level CSV has no corridor ID and no coordinates, so chainage values cannot be clustered into valid spatial hotspots.
- Weekday coding in the raw file is inconsistent across years (2013 to 2018 vs 2022 to 2023). V4 derives weekday from the parsed date and reports mismatches.
- The highway file contains crash events only, so it supports severity classification among recorded crashes, not crash-occurrence or hotspot prediction.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Confirms artifacts load; lists supported cities |
| `GET /api/overview` | Dataset and model diagnostics |
| `POST /api/forecast` | Per-cell crash-risk forecast (scenario or live Open-Meteo weather) |
| `POST /api/explain` | SHAP-based explanation for a selected cell |
| `GET /api/v4/status` | Which V4 outputs exist |
| `GET /api/v4/...` | V4 highway and Bengaluru research outputs (once the V4 pipeline has been run) |

Example forecast body:

```json
{
  "city": "Bangalore",
  "date": "2026-11-13",
  "hour": 12,
  "horizon": 6,
  "weather_mode": "scenario",
  "weather": "rain",
  "density": "medium"
}
```

Map tiles are CARTO dark tiles with OpenStreetMap attribution (no API key). Check the provider's usage terms before any production deployment.

---

## Project structure

```
api/                 FastAPI app (api/main.py)
frontend/            React + Vite + React-Leaflet dashboard
v4_pipeline/         India highway + Bengaluru data pipeline, severity models
artifacts/           Trained city model + context (models.joblib, context.joblib) and metrics
artifacts_benchmark/ Benchmark metrics and validation report
artifacts_v4/        Severity metrics and comparison tables
data/                Bundled city dataset, planted benchmark, V4 audit files
tests/               pytest suite
config.py            Grid size, split dates, thresholds, settings
data_prep.py, features.py, hotspots.py, modeling.py, train.py, predict.py
make_benchmark.py, validate_benchmark.py, diagnostics.py
app.py               Optional Streamlit app
```

## Limitations

- The bundled city dataset appears synthetic; its mapped cells are not verified real-world blackspots.
- Highway records contain no GPS coordinates or corridor identifiers.
- No traffic-volume or exposure denominator is available.
- The severity model predicts injury severity among recorded crashes, not whether a crash will occur.
- Bengaluru data is aggregated by police station and year.
- Saved `.joblib` models depend on library versions; retrain with `python train.py` if loading fails.

## Future work

- Obtain real geocoded crash records with corridor identifiers and traffic exposure.
- Route-aware spatial hotspot analysis on that data.
- Calibrate severity probabilities and improve severe-case detection.
- Integrate Bengaluru station-level analytics into the dashboard.

## Scope

RoadWatch is an academic research and development project. It is not an operational crash-prediction, emergency-response or road-safety decision system.
