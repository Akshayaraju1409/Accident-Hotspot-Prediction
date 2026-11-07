"""Central configuration (design doc: 'Configuration: grid size, thresholds, API settings')."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
# Override with env vars to run the SAME pipeline on another CSV / write to another folder, e.g. the
# planted-hotspot benchmark:  HOTSPOT_CSV=data/benchmark_planted.csv HOTSPOT_ART=artifacts_benchmark python train.py
ART = ROOT / os.environ.get("HOTSPOT_ART", "artifacts")
RAW_CSV = ROOT / os.environ.get("HOTSPOT_CSV", "data/indian_roads_dataset.csv")

# --- grid / time resolution -------------------------------------------------
CELL_DEG = 0.03          # ~3.3 km square cells (per-city grid)
BLOCK_H = 6              # hours per time block (4 blocks per day)
INDIA_LAT, INDIA_LON = (6.0, 37.5), (68.0, 98.0)

# --- time-based split (no random shuffling across time) ---------------------
FEATURE_END = "2022-12-31"   # hotspot labels / static cell features come ONLY from this window
TRAIN_START, TRAIN_END = "2023-01-01", "2023-12-31"   # model training rows
VAL_START, VAL_END = "2024-01-01", "2024-06-30"       # validation / calibration / thresholds
TEST_START = "2024-07-01"                              # test = everything after

# --- imbalance ---------------------------------------------------------------
N_BAGS = 5
SEVERE_LEVELS = ("fatal",)

# --- hotspot analysis --------------------------------------------------------
GI_K = 8                 # k nearest cells for Gi* neighbourhood
GI_PERMS = 499
FDR_Q = 0.05
DB_EPS_KM, DB_MIN = 1.5, 15

# --- risk engine -------------------------------------------------------------
HAZARDOUS_WEATHER = ("rain", "fog")
ALERT_QUANTILES = {"MEDIUM": 0.70, "HIGH": 0.92}   # of the validation score distribution
SEED = 42

# --- Open-Meteo (optional live forecast) ------------------------------------
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CITY_CENTRES = {
    "Bangalore": (12.97, 77.59), "Chandigarh": (30.73, 76.78), "Chennai": (13.08, 80.27),
    "Delhi": (28.61, 77.21), "Hyderabad": (17.38, 78.48), "Kolkata": (22.57, 88.36),
    "Mumbai": (19.08, 72.88), "Pune": (18.52, 73.86),
}
