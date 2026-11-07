
"""RoadWatch V4 highway preparation using published paper's coding table.
Never infer corridor identity from Accident_Location_A: it is urban/rural class.
"""
from pathlib import Path
import pandas as pd, numpy as np, json

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"v4"/"raw"
OUT=ROOT/"data"/"v4"/"processed"
OUT.mkdir(parents=True,exist_ok=True)

CODEBOOK={
 "Accident_Location_A":{"1":"Urban","2":"Rural","3":"Unallocated"},
 "Accident_Severity_C":{"1":"Fatal","2":"Grievous injury","3":"Minor injury","4":"No injury"},
 "Road_Feature_E":{"1":"Single lane","2":"Two lanes","3":"Three or more lanes without central divider","4":"Four or more lanes with central divider"},
 "Road_Condition_F":{"1":"Straight road","2":"Slight curve","3":"Sharp curve","4":"Flat road","5":"Gentle incline","6":"Steep incline","7":"Hump","8":"Dip"},
 "Weather_Conditions_H":{"1":"Fine","2":"Mist/Fog","3":"Cloud","4":"Light rain","5":"Heavy rain","6":"Hail/Sleet","7":"Snow","8":"Strong wind","9":"Dust storm","10":"Very hot","11":"Very cold","12":"Other extraordinary weather"},
 "Vehicle_Type_Involved_J_V1":{"1":"Car/Jeep/Van","2":"SUV","3":"Bus","4":"Mini bus","5":"Truck","6":"Two wheeler","7":"Three wheeler","8":"Cycle","9":"Pedestrian","10":"Tractor","11":"Unknown","12":"Animal","13":"Objects","14":"LCV","15":"MAV"},
 "Vehicle_Type_Involved_J_V2":{"1":"Car/Jeep/Van","2":"SUV","3":"Bus","4":"Mini bus","5":"Truck","6":"Two wheeler","7":"Three wheeler","8":"Cycle","9":"Pedestrian","10":"Tractor","11":"Unknown","12":"Animal","13":"Objects","14":"LCV","15":"MAV"},
}
def main():
    p=RAW/"indian_highway_accidents.csv"
    if not p.exists(): raise FileNotFoundError("Run python -m v4_pipeline.download_data first.")
    df=pd.read_csv(p)
    original=len(df)
    # Exact full-row duplicates only; preserve near-duplicates for review.
    exact_dup=df.duplicated(keep="first")
    clean=df.loc[~exact_dup].copy()
    # Normalize date/time without altering source values.
    clean["accident_date_parsed"]=pd.to_datetime(clean["Date"].astype(str).str.strip(),dayfirst=True,errors="coerce")
    clean["accident_time_parsed"]=pd.to_datetime(clean["Time_of_Accident"].astype(str).str.strip(),format="%H:%M",errors="coerce").dt.time
    clean["accident_datetime"]=pd.to_datetime(
        clean["accident_date_parsed"].dt.strftime("%Y-%m-%d")+" "+clean["Time_of_Accident"].astype(str).str.strip(),
        errors="coerce")
    clean["chainage_km_numeric"]=pd.to_numeric(clean["Accident_Location_A_Chainage_km"],errors="coerce")
    clean["source_day_of_week_code"]=pd.to_numeric(clean["Day_of_Week"],errors="coerce").astype("Int64")
    # Published article states 1=Sunday ... 7=Saturday. Preserve source code and flag
    # mismatches; do not silently overwrite either field.
    expected=((clean["accident_date_parsed"].dt.dayofweek+1)%7)+1
    clean["weekday_code_expected_sunday1"]=expected.astype("Int64")
    clean["weekday_from_date_iso"]=clean["accident_date_parsed"].dt.dayofweek.astype("Int64")
    clean["weekday_name_from_date"]=clean["accident_date_parsed"].dt.day_name()
    clean["weekday_code_mismatch"]=(
        clean["source_day_of_week_code"].notna() & expected.notna() &
        (clean["source_day_of_week_code"]!=expected)
    )
    # Add labels only where the published paper explicitly supplies a mapping.
    for col,mapping in CODEBOOK.items():
        if col in clean:
            clean[col+"_label"]=clean[col].map(lambda x: mapping.get(str(int(x)),"Unknown code") if pd.notna(x) and float(x).is_integer() else "Missing")
    # Causes_D is intentionally left as a raw category: the paper's variable
    # description is not sufficiently unambiguous to map this single CSV field.
    clean["Causes_D_code"]=clean["Causes_D"].astype("string")
    clean["Causes_D_label"]="Unmapped source category"
    # No corridor ID, route name, or GPS coordinates exist in this CSV.
    clean["corridor_id"]="NOT_PROVIDED"
    clean["spatially_locatable_by_chainage"]=False
    # Save exact unique records; don't discard near duplicates.
    clean.to_csv(OUT/"highway_accidents_clean.csv",index=False)
    # Possible duplicate groups are flagged for human review, not removed.
    key=["Date","Time_of_Accident","Accident_Location_A_Chainage_km","Accident_Location_A_Chainage_km_RoadSide","Accident_Severity_C"]
    near=clean[clean.duplicated(subset=key,keep=False)].sort_values(key)
    near.to_csv(OUT/"highway_possible_duplicate_review.csv",index=False)
    report={
      "source_rows":int(original),
      "exact_full_row_duplicates_removed":int(exact_dup.sum()),
      "exact_unique_rows":int(len(clean)),
      "invalid_parsed_date":int(clean["accident_date_parsed"].isna().sum()),
      "invalid_parsed_time":int(clean["accident_time_parsed"].isna().sum()),
      "invalid_chainage":int(clean["chainage_km_numeric"].isna().sum()),
      "weekday_mismatch_count_sunday_codebook":int(clean["weekday_code_mismatch"].sum()),
      "weekday_mismatch_rate_sunday_codebook":float(clean["weekday_code_mismatch"].mean()),
      "weekday_code_counts":{str(k):int(v) for k,v in clean["Day_of_Week"].value_counts(dropna=False).items()},
      "location_class_counts":{str(k):int(v) for k,v in clean["Accident_Location_A_label"].value_counts(dropna=False).items()},
      "severity_counts":{str(k):int(v) for k,v in clean["Accident_Severity_C_label"].value_counts(dropna=False).items()},
      "chainage_min":float(clean["chainage_km_numeric"].min()),
      "chainage_max":float(clean["chainage_km_numeric"].max()),
      "possible_duplicate_review_rows":int(len(near)),
      "corridor_ids_available":False,
      "gps_coordinates_available":False,
      "limitations":[
        "Accident_Location_A is coded urban/rural/unallocated, not a highway corridor identifier.",
        "The file has no route/corridor ID and no latitude/longitude. Chainage values cannot safely be clustered or mapped across routes.",
        "The Zenodo record describes four projects at dataset level, but the row-level CSV does not identify which project each accident belongs to.",
        "The associated paper's Table 3 supplies mappings for several coded fields; Causes_D remains unmapped because its relation to the paper's Causes-D1..D4 / nature-of-accident fields is ambiguous.",
        "The source weekday convention is inconsistent across time: records dated 2013-2018 match Monday=1..Sunday=7, while 2022-2023 match Sunday=1..Saturday=7. Use date-derived weekday for modelling and retain the original source code only for audit.",
        "This event-only sample supports severity classification among recorded crashes, not crash/no-crash forecasting."
      ]
    }
    (OUT/"highway_data_quality.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (OUT/"highway_codebook_partial.json").write_text(json.dumps({
      "source":"Table 3 of the associated published study, not a separately supplied official codebook",
      "source_url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12686657/",
      "mappings":CODEBOOK,
      "unmapped":["Causes_D","Accident_Location_A_Chainage_km_RoadSide"],
      "day_of_week":{"1":"Sunday","2":"Monday","3":"Tuesday","4":"Wednesday","5":"Thursday","6":"Friday","7":"Saturday"},
      "caution":"Use labels provisionally and cite the associated paper. Confirm ambiguous fields with the dataset authors before final research claims."
    },indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
