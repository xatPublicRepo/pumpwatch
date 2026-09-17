"""Fit the pump failure classifier on the current sensor snapshot."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sensor_readings.csv"
MODEL = ROOT / "models" / "pump_rf.joblib"


def load_params():
    params = {}
    for line in (ROOT / "params.yaml").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and ":" in line:
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
    return params


def main():
    params = load_params()
    frame = pd.read_csv(DATA)
    features = frame[["vibration_rms", "temp_c"]]
    labels = (frame["failure_score"] >= float(params["threshold"])).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=int(params["seed"])
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=int(params["seed"])
    )
    model.fit(x_train, y_train)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    print(json.dumps({"train_rows": len(x_train), "holdout_rows": len(x_test)}))


if __name__ == "__main__":
    main()
