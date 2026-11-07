
"""Normalize Bengaluru station-year aggregate CSVs without inventing point locations."""
from pathlib import Path
import pandas as pd, json, re
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data"/"v4"/"raw"; OUT=ROOT/"data"/"v4"/"processed"; OUT.mkdir(parents=True,exist_ok=True)
def num(v):
    if pd.isna(v): return None
    s=str(v).replace(",","").strip()
    try:return int(float(s))
    except:return None
def main():
    files=sorted(RAW.glob("bengaluru_btp_*.csv"))
    station_rows=[]; city_rows=[]; audit=[]
    for f in files:
        df=pd.read_csv(f)
        cols={str(c).strip():c for c in df.columns}
        if "Station" not in cols and "Year" in cols:
            for _,r in df.iterrows():
                city_rows.append({"year":num(r[cols["Year"]]),"fatal_crashes":num(r[cols.get("Fatal Road Accidents")]) if "Fatal Road Accidents" in cols else None,
                                  "killed_people":num(r[cols.get("Killed")]) if "Killed" in cols else None,
                                  "non_fatal_crashes":num(r[cols.get("Non-Fatal Road Accidents")]) if "Non-Fatal Road Accidents" in cols else None,
                                  "total_crashes":num(r[cols.get("Total")]) if "Total" in cols else None,"source_file":f.name})
            continue
        if "Station" not in cols: 
            audit.append({"file":f.name,"status":"skipped_no_station_column","columns":list(df.columns)}); continue
        # Detect year-prefixed measure columns and harmonize known naming variants.
        years=set()
        for c in cols:
            m=re.match(r"^(20\d{2})\s*[-–]",c)
            if m: years.add(int(m.group(1)))
        for _,r in df.iterrows():
            for y in sorted(years):
                def find_measure(patterns):
                    for c,orig in cols.items():
                        if c.startswith(str(y)) and any(re.search(p,c,re.I) for p in patterns):
                            return num(r[orig])
                    return None
                station_rows.append({
                  "year":y,"zone":r[cols["Zone"]] if "Zone" in cols else None,
                  "subdivision":r[cols["Sub-division"]] if "Sub-division" in cols else None,
                  "station":r[cols["Station"]],
                  "fatal_crashes":find_measure([r"\bfatal\b"]),
                  "non_fatal_crashes":find_measure([r"non.?fatal"]),
                  "total_crashes":find_measure([r"total\s+(cases|crashes)"]),
                  "killed_people":find_measure([r"killed"]),
                  "injured_people":find_measure([r"injured"]),
                  "source_file":f.name})
        audit.append({"file":f.name,"rows":len(df),"years":sorted(years),"status":"station_year_normalized"})
    st=pd.DataFrame(station_rows)
    if len(st):
        st.to_csv(OUT/"bengaluru_station_year.csv",index=False)
    city=pd.DataFrame(city_rows)
    if len(city): city.to_csv(OUT/"bengaluru_citywide_annual.csv",index=False)
    (OUT/"bengaluru_normalization_audit.json").write_text(json.dumps(audit,indent=2,default=str),encoding="utf-8")
    print(f"Station-year rows: {len(st)}; citywide annual rows: {len(city)}")
    print("Station aggregates are not GPS crash events. Check station naming/coverage changes before year-to-year comparisons.")
if __name__=="__main__": main()
