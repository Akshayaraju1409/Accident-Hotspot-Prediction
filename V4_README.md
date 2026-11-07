
# RoadWatch V4 — corrected India data pipeline

## Important source interpretation
The highway CSV has 13 columns. `Accident_Location_A` is the urban/rural/unallocated classification, NOT the corridor ID. The Zenodo landing page describes four highway projects at dataset level, but the supplied row-level CSV does not identify the project for each row. Therefore, chainage values cannot be grouped into valid spatial hotspots until row-level corridor identity and route geometry are obtained.

A related peer-reviewed paper provides a partial coding table for severity, location class, road feature, road condition, weather, vehicle type and weekday. This is a published-paper-derived partial codebook, not a separate official data dictionary. `Causes_D` and roadside coding remain unresolved. The associated paper's weekday mapping is not consistent with the raw file across years: 2013-2018 rows match Monday=1..Sunday=7, while 2022-2023 rows match Sunday=1..Saturday=7. V4 derives weekday from the parsed date for modelling, retains the source code, and reports mismatches.

## Run
```powershell
python -m pip install -r requirements.txt
python -m v4_pipeline.download_data
python -m v4_pipeline.prepare_highway
python -m v4_pipeline.prepare_bengaluru
python -m v4_pipeline.highway_hotspots
python -m v4_pipeline.train_severity
```

## Outputs
- `data/v4/processed/highway_accidents_clean.csv`
- `data/v4/processed/highway_possible_duplicate_review.csv`
- `data/v4/processed/highway_data_quality.json`
- `data/v4/processed/highway_codebook_partial.json`
- `data/v4/processed/highway_hotspot_candidates_status.json`
- `data/v4/processed/bengaluru_station_year.csv`
- `data/v4/processed/bengaluru_citywide_annual.csv`
- `data/v4/processed/bengaluru_normalization_audit.json`

## Model scope
The highway file contains observed crash events only. It supports severity classification among recorded crashes. It does not contain the full denominator of segment-time exposure / zero-crash intervals, so it cannot independently train a valid crash occurrence probability model. Do not interpret severity classification as future crash prediction.

Bengaluru station-year records support aggregate descriptive trends, not street-level point clustering or hourly prediction.

## External source for partial codebook
Associated study: https://pmc.ncbi.nlm.nih.gov/articles/PMC12686657/
Dataset record: https://zenodo.org/records/16946653
