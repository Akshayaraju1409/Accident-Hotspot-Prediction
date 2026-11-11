from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd, numpy as np, datetime as dt
from predict import Service, fetch_open_meteo, WeatherError
from api.severity import router as severity_router
app=FastAPI(title='RoadWatch API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
app.include_router(severity_router)   # /api/v4/severity/*
_s=None
def svc():
 global _s
 if _s is None: _s=Service()
 return _s
class ForecastRequest(BaseModel):
 city:str='Bangalore'; date:str; hour:int=Field(12,ge=0,le=23); horizon:int=Field(6,ge=1,le=6); weather_mode:str='scenario'; weather:str='rain'; density:str='medium'
@app.get('/api/health')
def health():
 try: return {'status':'ok','cities':svc().cities,'pipeline':'loaded'}
 except Exception as e: raise HTTPException(503,f'Prediction artifacts unavailable: {e}')
@app.get('/api/overview')
def overview():
 s=svc(); return {'cities':s.cities,'metrics':s.metrics,'dataset_rows':int(len(s.crashes)),'hotspot_cells':int(len(s.hotspots)),'data_note':'The bundled city dataset appears synthetic and does not establish verified real-world hotspot locations.'}
@app.post('/api/forecast')
def forecast(q:ForecastRequest):
 s=svc()
 if q.city not in s.cities: raise HTTPException(400,'Unknown city')
 start=pd.Timestamp(dt.datetime.combine(dt.date.fromisoformat(q.date),dt.time(q.hour)))
 try:
  w=fetch_open_meteo(q.city,start,q.horizon) if q.weather_mode=='live' else q.weather
  df=s.forecast(q.city,start,w,q.density,q.horizon)
 except WeatherError as e: raise HTTPException(502,str(e))
 except Exception as e: raise HTTPException(400,str(e))
 rows=[]
 for r in df.to_dict(orient='records'):
  for k,v in list(r.items()):
   if isinstance(v,(pd.Timestamp,dt.datetime,dt.date)): r[k]=v.isoformat()
   elif isinstance(v,(np.integer,)): r[k]=int(v)
   elif isinstance(v,(np.floating,)): r[k]=None if not np.isfinite(v) else float(v)
   elif isinstance(v,(np.bool_,)): r[k]=bool(v)
  rows.append(r)
 return {'city':q.city,'start':start.isoformat(),'weather':w if isinstance(w,str) else 'live forecast','density':q.density,'rows':rows}
class ExplainRequest(BaseModel): city:str; timestamp:str; cell_id:int; weather:str='rain'; density:str='medium'
@app.post('/api/explain')
def explain(q:ExplainRequest):
 try:
  x=svc().explain(q.city,pd.Timestamp(q.timestamp),q.cell_id,q.weather,q.density,top=8)
  return {'reasons':x.replace({np.nan:None}).to_dict(orient='records')}
 except Exception as e: raise HTTPException(400,str(e))


# RoadWatch V4 research-data endpoints. These are deliberately separate from
# the legacy V3 prediction endpoints until V4 data and geography are validated.
from pathlib import Path as _Path
import json as _json
_V4_ROOT = _Path(__file__).resolve().parents[1]
_V4_PROCESSED = _V4_ROOT / "data" / "v4" / "processed"
_V4_ARTIFACTS = _V4_ROOT / "artifacts_v4"

@app.get("/api/v4/status")
def v4_status():
    return {
        "version": "4.0",
        "highway_cleaned": (_V4_PROCESSED / "highway_accidents_clean.csv").exists(),
        "highway_segments": (_V4_PROCESSED / "highway_chainage_segments_1km.csv").exists(),
        "highway_hotspots": (_V4_PROCESSED / "highway_hotspot_candidates.csv").exists(),
        "bengaluru_schema_audit": (_V4_PROCESSED / "bengaluru_btp_schema_audit.json").exists(),
        "severity_model": (_V4_ARTIFACTS / "highway_severity_baseline.joblib").exists(),
        "severity_v2_model": (_V4_ARTIFACTS / "highway_severity_v2_binary_best.joblib").exists(),
        "severity_analysis": (_V4_ARTIFACTS / "analysis" / "severity_analysis.json").exists(),
        "note": "V4 highway chainage results are not GPS points; Bengaluru BTP data are station-level aggregates."
    }

@app.get("/api/v4/highway/hotspots")
def v4_highway_hotspots(limit: int = 500):
    p = _V4_PROCESSED / "highway_hotspot_candidates.csv"
    if not p.exists():
        status_file = _V4_PROCESSED / "highway_hotspot_candidates_status.json"
        if status_file.exists():
            return _json.loads(status_file.read_text(encoding="utf-8"))
        raise HTTPException(404, "Run the V4 data preparation and hotspot assessment first.")
    df = pd.read_csv(p).sort_values(["hotspot_candidate", "accident_count"], ascending=[False, False]).head(max(1, min(limit, 2000)))
    return {"mode": "chainage_segments", "map_coordinates_available": False,
            "rows": df.replace({np.nan: None}).to_dict(orient="records"),
            "warning": "Exploratory chainage segments; not verified GPS locations or official blackspots."}

@app.get("/api/v4/highway/metrics")
def v4_highway_metrics():
    # Prefer the corrected V2 binary model; fall back to the older four-class baseline if V2 has not been run.
    for name in ("highway_severity_v2_metrics.json", "highway_severity_metrics.json"):
        p = _V4_ARTIFACTS / name
        if p.exists():
            return _json.loads(p.read_text(encoding="utf-8"))
    raise HTTPException(404, "Run python -m v4_pipeline.severity_v2 first.")

@app.get("/api/v4/bengaluru/audit")
def v4_bengaluru_audit():
    p = _V4_PROCESSED / "bengaluru_btp_schema_audit.json"
    if not p.exists():
        raise HTTPException(404, "Run the V4 Bengaluru data audit first.")
    return _json.loads(p.read_text(encoding="utf-8"))
