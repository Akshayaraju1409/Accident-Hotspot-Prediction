
"""Refuse invalid spatial clustering until route IDs / geometry are available."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"v4"/"processed"
def main():
    clean=OUT/"highway_accidents_clean.csv"
    if not clean.exists():
        raise FileNotFoundError("Run python -m v4_pipeline.prepare_highway first.")
    result={
      "status":"blocked_pending_route_identity",
      "message":"The CSV contains chainage but no row-level corridor ID or GPS coordinates. Accident_Location_A is urban/rural class, not corridor ID. Clustering chainage across the dataset would mix unrelated roads and produce invalid hotspots.",
      "required_inputs":["row-level corridor/route identifier","route geometry for each corridor","validated chainage-to-coordinate calibration or network adjacency","traffic exposure/AADT if interpreting risk rather than raw crash concentration"],
      "available_next_step":"Continue severity classification and Bengaluru station-level trend analysis; acquire route metadata before spatial hotspot detection."
    }
    (OUT/"highway_hotspot_candidates_status.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
