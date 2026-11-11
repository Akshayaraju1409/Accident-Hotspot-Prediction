# Severity upgrade v2: accuracy experiment, UI upgrade, push checklist

Unzip over your project root (the folder with `api/`, `v4_pipeline/`, `frontend/`, `tests/`). Overwrites `.gitignore`,
`requirements.txt`, `severity_v2.py`, `severity_analysis.py`, `api/*.py`, `frontend/src/*`; adds `severity_features.py`,
`severity_improve.py` and `tests/test_severity_features.py`.

**API port:** `frontend/src/main.jsx` now defaults to `http://localhost:8001/api` (what you used). To use another port, create `frontend/.env` with `VITE_API_URL=http://localhost:8000/api`.

## 1. Install and test
```powershell
pip install -r requirements.txt
python -m pytest -q tests                      # expect 30 passed
```

## 2. Retrain the baseline, then try the accuracy improvement
```powershell
python -m v4_pipeline.severity_v2              # baseline; first line must say: Rows: source 4053, usable 4053, dropped 0
python -m v4_pipeline.severity_improve         # experiment only: overwrites nothing
```
Read `artifacts_v4\improve\improve_report.md` and the last lines printed:
- `Test ROC-AUC: V2 x -> improved y (paired delta ..., 95% CI ...)`
- The verdict line says whether the gain is distinguishable from noise.

Expect a small change (a few AUC points at most). The data has 11 inputs and no speed, traffic volume or road geometry, so
there is a ceiling. If the gain is within noise, adopting is still defensible (more training data, calibrated scores), but
describe the accuracy as essentially unchanged.

To use the improved model:
```powershell
python -m v4_pipeline.severity_improve --adopt # old model is backed up as ...before_improve.joblib
python -m v4_pipeline.severity_analysis        # REQUIRED after adopting: ablation, errors, README snippet
```
Skip `--adopt` to keep the baseline, then still run `severity_analysis`.

## 3. Run the app
```powershell
python -m uvicorn api.main:app --port 8001     # use the port you configured in frontend\src\main.jsx
cd frontend ; npm run dev                      # second terminal; open http://localhost:5173
```
Check in the browser:
1. A yellow "Demo data" banner on Overview / Hotspot explorer / Risk analytics.
2. "Crash severity" tab: weak-model banner, 4 scenario buttons, form, then the result with a "% of similar crashes that were severe" headline.
3. Below: four charts (ablation, catching severe crashes, calibration, model comparison).

## 4. Push checklist: you can push when ALL of these are true
- [ ] `python -m pytest -q tests` passes.
- [ ] `severity_v2` printed `dropped 0` and you ran `severity_analysis` after the last model change.
- [ ] README.md metrics replaced with the generated section: open `artifacts_v4\analysis\readme_severity_section.md` and paste it over the
      old block (from `A chronological train-validation-test strategy...` down to the sentence before `### 3. Bengaluru Accident Analysis`).
      The old text still quotes ROC-AUC 0.5985 and "99 of 153", which came from the buggy run.
- [ ] No model binaries are tracked (they are 17-47 MB each, 200 MB+ in total):
```powershell
git status --short
git ls-files | Select-String "\.joblib"        # must print nothing
```
      If it lists files: `git rm --cached <each file>` then commit. If they were pushed in an EARLIER commit they stay in history
      (fine under GitHub's 100 MB per-file limit, but the repo stays large).
- [ ] The two app tabs you changed open without errors in the browser.
- [ ] You push to the feature branch and open a pull request instead of pushing straight to main:
```powershell
git checkout -b feature-severity-v2            # skip if already on it
git add -A
git status                                     # skim the list: no .venv, node_modules, .joblib, raw data
git commit -m "Fix severity date parsing; add analysis, calibrated ensemble option, API and UI"
git push -u origin feature-severity-v2
```
After a fresh clone the app needs its models: run `python train.py` (city model, about 6 min) and `python -m v4_pipeline.severity_v2`.

## 5. What is and is not claimed
Severity of a RECORDED crash only. Weak discrimination (AUC well below 0.7). City-tab forecasts are on synthetic data.
