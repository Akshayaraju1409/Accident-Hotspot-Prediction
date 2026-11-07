# Accident hotspot & weather-risk forecast (v2: cell x time-block formulation)

Built from the design doc, workflow and proposal, adapted to `data/indian_roads_dataset.csv`
(20,000 crashes, 8 Indian cities, 2022-01 to 2025-04).

## Run
```bash
pip install -r requirements.txt
python train.py             # ~6 min: hotspots, 5 models + balanced ensemble, calibration, LOCO, SHAP -> artifacts/
python diagnostics.py       # within-city chance test, appended to artifacts/metrics.json
streamlit run app.py        # dashboard on the real CSV
pytest -q tests             # 12 tests incl. planted-hotspot recovery

# Method validation on a dataset where hotspots really exist (see next section)
python make_benchmark.py                                   # -> data/benchmark_planted.csv + benchmark_truth.json
HOTSPOT_CSV=data/benchmark_planted.csv HOTSPOT_ART=artifacts_benchmark python train.py
HOTSPOT_CSV=data/benchmark_planted.csv HOTSPOT_ART=artifacts_benchmark python diagnostics.py
python validate_benchmark.py                               # ground-truth recovery -> artifacts_benchmark/validation.json
HOTSPOT_ART=artifacts_benchmark streamlit run app.py       # dashboard with the planted hotspots
```

## Read this first: what this dataset can and cannot support
`indian_roads_dataset.csv` looks synthetic and has **no spatial or temporal crash structure**:
- Crash locations are uniform inside each city's box (grid chi-square p = 0.09-0.97 per city), hours/days
  are uniform, `cause` is independent of weather, and nothing predicts fatal vs non-fatal (AUC ~0.50).
- After fixing the Gi* bug (below) the detectors find essentially nothing: Gi* (BH-FDR) flags 1 of 1,137
  cells, DBSCAN 13, both consistent with chance. Moran's I is ~0.
- The ensemble's overall PR-AUC lift (~1.7x) is **only the city effect** (cities with fewer cells have more
  crashes per cell; city-only oracle 1.64). Within each city the lift is 0.97-1.09, inside the shuffled null band.
- `risk_score` is a formula of weather, traffic density and peak hour (R2 0.89). Weather-driven alert levels
  are real model behaviour; where-in-the-city differences are noise.

## Bug fixed: Gi* used the wrong statistic
`hotspots.py` read `G_Local.Zs`, which is not the Gi* z-score and understated it roughly 3x (max z ~1.6 where the
permutation and analytic Gi* both give ~4.6). Gi* therefore could not flag anything, even obvious clusters.
It now uses `z_sim`; `tests/test_benchmark.py::test_gi_star_uses_a_real_z_score` guards it.

## Proof the pipeline works when hotspots exist: planted-hotspot benchmark
`make_benchmark.py` keeps every real row and adds 4,124 extra crashes around 32 KNOWN hotspot centres
(4 per city, persistent over the whole period, skewed to peak hours and rain/fog). Because the truth is known:

| Check | Original CSV (no hotspots) | Planted benchmark |
|---|---|---|
| Gi* hot cells (FDR 5%) | 1 of 1,137 | 60 |
| DBSCAN hot cells | 13 | 104 |
| Planted hotspots found by DBSCAN (within 1 cell) | n/a | 32 of 32 |
| Planted hotspots found by Gi* (within 1 cell) | n/a | 21 of 32 (weak ones ~50%) |
| Share of FUTURE crashes in the past top-10% cells (random = 10%, oracle ceiling 24.8%) | 10.5% | 23.3% |
| Within-city model lift, 8 cities (null p95 ~1.1) | 0.97-1.09 | 1.46-3.30 |
| Test PR-AUC lift of final ensemble | 1.74 (city effect only) | 2.82 |

The benchmark is a method check, not evidence about real Indian roads: it shows the code detects and forecasts
spatial hotspots when they exist and stays quiet when they do not. Hotspot features come from the 2022 window only,
and evaluation is on later, unseen months.

## Real Indian data worth trying (checked Nov 2026)
Crash-level data with coordinates is rarely public in India; most open data is aggregated.
- **NHAI highway crash records (Zenodo 7773156, CC-BY 4.0)**: crash records for NH-9 Pune-Solapur (2013-18) and
  NH-2 Barwa-Adda-Panagarh (2015-19), ~250 kB. Highway chainage rather than city grids, so it needs a 1-D
  (km-segment) version of the grid; best real option for severity/segment hotspots.
- **Delhi Traffic Police black spots**: Delhi's traffic police publish identified black spots (153 listed for 2025
  in press coverage) plus annual crash reports and 2014-23 statistics on OpenCity (data.opencity.in). Useful as
  external ground truth to validate hotspots found from any coordinates you can obtain.
- **Bengaluru Traffic Police station-wise crashes and fatalities, 2020-22 (OpenCity)**: aggregated by station, so
  you would use polygon/station units instead of grid cells.
- **NCRB ADSI / MoRTH via data.gov.in, OpenCity, India Data Portal, AIKosh**: state, city, road-class and
  place-of-occurrence counts. Aggregated only: good for context features, not point hotspots.
- **Top-10 national blackspots (dataful.in)** and **iRAD (Integrated Road Accident Database)**: iRAD is the
  official source and powers police-station reports, but crash-level exports are not openly downloadable.
Free APIs the pipeline can use: Open-Meteo (already used for live weather); OpenStreetMap/Overpass for road class,
junctions and signals as static features (not wired in yet).

## Design -> code
| Module | File |
|---|---|
| M1/M2 clean, grid, cell x 6-h block counts | `data_prep.py` |
| M3 Gi* (BH-FDR) + DBSCAN + Moran's I, method comparison | `hotspots.py` |
| M4 features: causal rolling history + static features from 2022 only | `features.py` |
| M5/M6/M7 split, models, calibration, metrics, bootstrap CI | `modeling.py`, `train.py` |
| M8/M9/M10 forecast (1-6 h), risk engine, SHAP | `predict.py` |
| M11 Streamlit + Folium dashboard | `app.py` |

Split: static hotspot features from 2022; train 2023; validate/calibrate/threshold 2024-H1; test 2024-07 to 2025-04.
Imbalance: balanced bagging (5 negative-undersampled LightGBM bags; ensemble variance = uncertainty); SMOTE only as ablation.
Alert = (P(crash) relative to the city's average cell) x (conditions risk from weather, traffic, hour) vs thresholds
(70th / 92nd percentile of the validation score distribution; stored in `models.joblib`).

## Changes from the design doc
- No weather exists for no-crash cell-hours in this data, so weather enters the alert through the conditions
  model, not the crash-occurrence model. Weather scenario is chosen in the UI, or fetched from Open-Meteo
  ("Auto"; no forecast -> no alert). The live API call was only tested offline (mock payload; sandbox has no internet).
- P(severe | crash) is shown but flagged as no-skill.
- Cross-city test is leave-one-city-out (no second dataset).
- Not built: PostgreSQL/PostGIS (files are used), scheduled retraining, OSM road features.
