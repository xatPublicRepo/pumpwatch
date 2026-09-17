# pumpwatch

Failure scoring for the pump fleet. Sensor readings arrive as a daily snapshot,
`src/train.py` fits the classifier, `src/evaluate.py` scores it against the
holdout window, and `src/report.py` summarises the snapshot that is currently
checked out.

## Layout

- `src/` importable code
- `notebooks/` exploration only, nothing in `src/` imports from here
- `configs/` sweep and experiment configuration
- `params.yaml` the knobs for a run
- `data/` the sensor snapshot, refreshed daily
- `models/` the fitted classifier
- `artifacts/` whatever a run writes out

## Running the report

    python3 src/report.py

It reads `data/sensor_readings.csv` and `params.yaml` and writes
`artifacts/metrics.json`.
