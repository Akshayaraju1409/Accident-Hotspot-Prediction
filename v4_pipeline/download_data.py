
"""Download documented public datasets for RoadWatch V4.

The highway CSV is an event-level dataset. Bengaluru BTP data are station-level
aggregates and must never be treated as point-level crash coordinates.
"""
from pathlib import Path
import json, requests, sys

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"v4"/"raw"
RAW.mkdir(parents=True,exist_ok=True)
ZENODO_API="https://zenodo.org/api/records/16946653"
OPENCITY_API="https://data.opencity.in/api/3/action/package_show?id=bengaluru-road-crashes-data"

def download(url,path):
    r=requests.get(url,timeout=60); r.raise_for_status()
    path.write_bytes(r.content)
    print(f"Downloaded {path.name}: {len(r.content):,} bytes")

def main():
    meta=requests.get(ZENODO_API,timeout=30); meta.raise_for_status()
    record=meta.json()
    files=record.get("files",[])
    match=next((f for f in files if f["key"].lower().endswith(".csv")),None)
    if not match: raise RuntimeError("Could not find the highway CSV in Zenodo record.")
    download(match["links"]["self"],RAW/"indian_highway_accidents.csv")

    # CKAN resource discovery; download CSV resources and retain their original names.
    try:
        r=requests.get(OPENCITY_API,timeout=30); r.raise_for_status()
        result=r.json()
        if not result.get("success"): raise RuntimeError("OpenCity CKAN did not return success.")
        resources=result["result"].get("resources",[])
        csvs=[x for x in resources if x.get("format","").lower()=="csv" or x.get("url","").lower().endswith(".csv")]
        if not csvs: print("No direct CSV resources were returned. See the OpenCity page and download manually.")
        for i,res in enumerate(csvs,1):
            title="".join(c if c.isalnum() or c in "-_" else "_" for c in res.get("name") or f"btp_resource_{i}")
            path=RAW/f"bengaluru_btp_{i}_{title[:60]}.csv"
            download(res["url"],path)
    except Exception as e:
        print(f"OpenCity automatic download unavailable: {e}")
        print("Download CSV resources manually from https://data.opencity.in/dataset/bengaluru-road-crashes-data")
    (RAW/"download_manifest.json").write_text(json.dumps({
        "highway_record":ZENODO_API,
        "bengaluru_record":OPENCITY_API,
        "note":"Preserve original source files; do not merge station aggregates with event-level records."
    },indent=2),encoding="utf-8")
    print(f"Raw datasets saved under {RAW}")

if __name__=="__main__": main()
