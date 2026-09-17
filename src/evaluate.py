"""Score the fitted classifier against the holdout window."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sensor_readings.csv"
MODEL = ROOT / "models" / "pump_rf.joblib"


def main():
    frame = pd.read_csv(DATA).tail(20000)
    model = joblib.load(MODEL)
    scores = model.predict_proba(frame[["vibration_rms", "temp_c"]])[:, 1]
    truth = (frame["failure_score"] >= 0.62).astype(int)
    print(json.dumps({"roc_auc": round(float(roc_auc_score(truth, scores)), 4)}))


if __name__ == "__main__":
    main()
