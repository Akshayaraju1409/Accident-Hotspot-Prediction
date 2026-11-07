
"""Temporal-holdout severity classification among observed crashes only."""
from pathlib import Path
import pandas as pd, numpy as np, json, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"data"/"v4"/"processed"; ART=ROOT/"artifacts_v4"
def main():
    p=OUT/"highway_accidents_clean.csv"
    if not p.exists(): raise FileNotFoundError("Run python -m v4_pipeline.prepare_highway first.")
    df=pd.read_csv(p)
    df["date"]=pd.to_datetime(df["Date"].astype(str),dayfirst=True,errors="coerce")
    df["time_minutes"]=pd.to_datetime(df["Time_of_Accident"].astype(str),format="%H:%M",errors="coerce").dt.hour*60 + pd.to_datetime(df["Time_of_Accident"].astype(str),format="%H:%M",errors="coerce").dt.minute
    df["month"]=df["date"].dt.month
    df["year"]=df["date"].dt.year
    df["weekday_from_date"]=df["date"].dt.dayofweek.astype("string")
    target="Accident_Severity_C"
    df=df.dropna(subset=["date",target]).sort_values("date").copy()
    # Chronological holdout: latest 20% of unique dates are test period.
    unique_dates=np.sort(df["date"].dropna().unique())
    if len(unique_dates)<5: raise ValueError("Not enough distinct dates for temporal holdout.")
    cut=unique_dates[max(1,int(len(unique_dates)*0.8))]
    tr=df[df["date"]<cut].copy(); te=df[df["date"]>=cut].copy()
    if tr.empty or te.empty: raise ValueError("Temporal split produced an empty partition.")
    features=["Accident_Location_A","Accident_Location_A_Chainage_km",
              "Accident_Location_A_Chainage_km_RoadSide","Causes_D","Road_Feature_E",
              "Road_Condition_F","Weather_Conditions_H","Vehicle_Type_Involved_J_V1",
              "Vehicle_Type_Involved_J_V2","weekday_from_date_iso","time_minutes","month"]
    features=[c for c in features if c in df.columns]
    categorical=[c for c in features if c not in ["Accident_Location_A_Chainage_km","time_minutes","month"]]
    numeric=[c for c in features if c in ["Accident_Location_A_Chainage_km","time_minutes","month"]]
    Xtr=tr[features].copy(); Xte=te[features].copy()
    for c in categorical:
        Xtr[c]=Xtr[c].astype(object).where(Xtr[c].notna(),np.nan)
        Xte[c]=Xte[c].astype(object).where(Xte[c].notna(),np.nan)
    ytr=tr[target].astype(str); yte=te[target].astype(str)
    transformers=[]
    if categorical:
        transformers.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),categorical))
    if numeric:
        transformers.append(("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),numeric))
    prep=ColumnTransformer(transformers)
    model=Pipeline([("prep",prep),("rf",RandomForestClassifier(
        n_estimators=600,min_samples_leaf=3,class_weight="balanced_subsample",
        max_features="sqrt",random_state=42,n_jobs=-1))])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    labels=sorted(set(ytr)|set(yte))
    report=classification_report(yte,pred,labels=labels,output_dict=True,zero_division=0)
    ART.mkdir(exist_ok=True); joblib.dump(model,ART/"highway_severity_baseline.joblib")
    metrics={"target":"Accident_Severity_C","task":"severity classification among observed crashes (not crash occurrence prediction)",
      "model":"RandomForestClassifier","split":"chronological 80/20 by distinct accident date",
      "cutoff_date":str(pd.Timestamp(cut).date()),"train_rows":len(tr),"test_rows":len(te),
      "train_date_range":[str(tr.date.min().date()),str(tr.date.max().date())],
      "test_date_range":[str(te.date.min().date()),str(te.date.max().date())],
      "features":features,"balanced_accuracy":float(balanced_accuracy_score(yte,pred)),
      "macro_f1":float(f1_score(yte,pred,average="macro")),
      "labels":labels,"confusion_matrix":confusion_matrix(yte,pred,labels=labels).tolist(),
      "classification_report":report,
      "limitations":["Event-only dataset; cannot estimate probability of a crash in a segment-time interval.",
        "Row-level highway corridor identity is absent; geographic transfer validation is not possible.",
        "Source weekday coding changes convention across time; date-derived weekday is used for modelling and raw source code is retained for audit.",
        "Category codes are preserved as categories; see highway_codebook_partial.json for mappings supported by the associated paper.",
        "Potential near-duplicate records are retained for review; only exact full-row duplicates are removed."]}
    (ART/"highway_severity_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k!="classification_report"},indent=2))
if __name__=="__main__": main()
