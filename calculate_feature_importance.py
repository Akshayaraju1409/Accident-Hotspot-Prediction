
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from v4_pipeline.severity_v2 import (
    prepare_data,
    clean_features,
    CAT_FEATURES,
    NUM_FEATURES,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts_v4" / "highway_severity_v2_binary_best.joblib"
OUTPUT_DIR = ROOT / "artifacts_v4"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the exact trained V2 model
saved = joblib.load(MODEL_PATH)
model = saved["model"]
threshold = saved["threshold"]

# Recreate the same chronological test split
df, train, val, test = prepare_data()

cat_cols = [c for c in CAT_FEATURES if c in df.columns]
num_cols = [c for c in NUM_FEATURES if c in df.columns]

X_test = clean_features(test, cat_cols, num_cols)
y_test = test["severe_binary"].astype(int)

# Calculate permutation importance using average precision.
# This measures how much ranking performance drops when a feature is shuffled.
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

# Save results
csv_path = OUTPUT_DIR / "highway_feature_importance.csv"
importance.to_csv(csv_path, index=False)

# Plot results
plot_data = importance.sort_values("importance_mean")

plt.figure(figsize=(10, 6))
plt.barh(
    plot_data["feature"],
    plot_data["importance_mean"],
    xerr=plot_data["importance_std"]
)
plt.xlabel("Decrease in Average Precision")
plt.ylabel("Feature")
plt.title("RoadWatch V2 - Permutation Feature Importance")
plt.tight_layout()

plot_path = OUTPUT_DIR / "highway_feature_importance.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print("\nFEATURE IMPORTANCE RESULTS")
print(importance.to_string(index=False))
print(f"\nCSV saved to: {csv_path}")
print(f"Chart saved to: {plot_path}")
print(f"Model threshold: {threshold}")
print(f"Test records analysed: {len(X_test)}")
