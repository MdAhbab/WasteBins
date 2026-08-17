"""
External validation on real IoT telemetry.
==========================================

Answers reviewer 2 point 7 in full: the earlier submission referred to
"validation on public telemetry from real devices" without naming the dataset,
the devices, the duration, or the comparison metric. All of that is reported
here, computed from the file rather than asserted.

Dataset
-------
"Environmental Sensor Telemetry Data" (Gary Stafford, Kaggle, CC0):
https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k

Three physically separate ESP8266 nodes reporting carbon monoxide, LPG, smoke,
temperature, humidity, light and motion at roughly one-second cadence in July
2020. Column set: ``ts, device, co, humidity, light, lpg, motion, smoke, temp``.

Why this dataset, stated plainly
--------------------------------
No public dataset carries bin *fill* level alongside gas, temperature and
humidity, so no real dataset can validate the full pipeline end to end.  What
this one does provide is the genuine noise, drift, dropout and device-to-device
heterogeneity of low-cost IoT hardware on exactly the hazard modalities the
system relies on.  It therefore validates the component most exposed to real
sensor behaviour — forward hazard prediction from a noisy recent past — and the
manuscript claims nothing beyond that.

What is measured
----------------
The *same* feature contract, the *same* forward-labelling rule and the *same*
calibrated classifier used on the simulation are applied unchanged. Reported
against three references a reader can interpret:

  * the current gas reading used directly as a score (persistence),
  * a logistic regression on the same features (is the non-linearity needed?),
  * the base rate (is the Brier score actually informative?).

Validation is a temporal hold-out *within each device*, and a leave-one-device-out
run additionally tests transfer to a node never seen in training — the honest
analogue of installing a new bin.

Obtaining the data
------------------
    kaggle datasets download -d garystafford/environmental-sensor-data-132k \\
        -p experiments/realdata --unzip
or place ``iot_telemetry_data.csv`` in ``experiments/realdata/`` or the
repository's parent directory.

Run:  python exp_realdata.py [path/to/iot_telemetry_data.csv]
Out:  results/realdata.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import hpo as HPO  # noqa: E402

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

RESAMPLE = "1min"      # devices report ~1 Hz; 1-minute bins keep the file tractable
ROLL = 10              # rolling window, matching the deployed feature contract
HORIZON = 30           # forward horizon in resampled steps (30 minutes)
HAZARD_PCTL = 95       # per-device percentile defining a gas hazard
MIN_ROWS = 200


def resolve_csv(argv: List[str]) -> pathlib.Path:
    if len(argv) > 1 and argv[1].lower().endswith(".csv"):
        return pathlib.Path(argv[1])
    if os.environ.get("REALDATA_CSV"):
        return pathlib.Path(os.environ["REALDATA_CSV"])
    for candidate in (HERE / "realdata" / "iot_telemetry_data.csv",
                      HERE.parent.parent / "iot_telemetry_data.csv",
                      HERE.parent / "iot_telemetry_data.csv"):
        if candidate.exists():
            return candidate
    return HERE / "realdata" / "iot_telemetry_data.csv"


def load(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[!] Dataset not found at {path}\n"
              f"    See this file's docstring for one-line download instructions.")
        sys.exit(2)
    frame = pd.read_csv(path)
    frame["_dt"] = pd.to_datetime(frame["ts"], unit="s", errors="coerce")
    return frame.dropna(subset=["_dt"])


def describe_source(frame: pd.DataFrame, path: pathlib.Path) -> Dict:
    """Everything a reader needs to identify and re-obtain the data."""
    span = frame["_dt"].max() - frame["_dt"].min()
    per_device = frame.groupby("device").agg(
        rows=("device", "size"),
        first=("_dt", "min"),
        last=("_dt", "max"),
    )
    return {
        "name": "Environmental Sensor Telemetry Data (Gary Stafford, Kaggle, CC0)",
        "url": "https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k",
        "file": path.name,
        "raw_rows": int(len(frame)),
        "devices": int(frame["device"].nunique()),
        "device_ids": sorted(frame["device"].unique().tolist()),
        "collection_start": frame["_dt"].min().isoformat(),
        "collection_end": frame["_dt"].max().isoformat(),
        "duration_days": round(span.total_seconds() / 86400.0, 2),
        "median_sampling_interval_s": round(float(
            frame.sort_values("_dt")["_dt"].diff().dt.total_seconds().median()), 3),
        "per_device": {
            str(device): {
                "rows": int(row["rows"]),
                "first": row["first"].isoformat(),
                "last": row["last"].isoformat(),
            }
            for device, row in per_device.iterrows()
        },
        "channels_used": {
            "gas": "smoke (hazard modality)",
            "temp": "temp",
            "humidity": "humidity",
        },
        "channels_available_unused": ["co", "lpg", "light", "motion"],
    }


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling features and the forward hazard label, per device.

    Deliberately mirrors ``wastebins_core.features``: current value, rolling
    mean, rolling standard deviation and a per-step trend for each channel, plus
    cyclical time. Reimplementing it here rather than importing is not
    duplication for its own sake — the real stream is resampled and has no
    bin-fill channel, so the shapes differ — but the *statistics* are identical,
    which is what makes the comparison meaningful.
    """
    blocks = []
    for device, group in frame.groupby("device"):
        series = group.set_index("_dt").sort_index()
        numeric = pd.DataFrame({
            "gas": pd.to_numeric(series["smoke"], errors="coerce"),
            "temp": pd.to_numeric(series["temp"], errors="coerce"),
            "humidity": pd.to_numeric(series["humidity"], errors="coerce"),
        })
        resampled = numeric.resample(RESAMPLE).mean().interpolate().dropna()
        if len(resampled) < ROLL + HORIZON + 50:
            continue

        features: Dict[str, object] = {}
        for channel in ("gas", "temp", "humidity"):
            values = resampled[channel]
            features[channel] = values
            features[f"mean_{channel}"] = values.rolling(ROLL, min_periods=1).mean()
            features[f"std_{channel}"] = values.rolling(ROLL, min_periods=2).std().fillna(0.0)
            features[f"trend_{channel}"] = (values - values.shift(ROLL)).fillna(0.0) / ROLL
            features[f"ewma_{channel}"] = values.ewm(halflife=3).mean()

        built = pd.DataFrame(features)
        hour = resampled.index.hour + resampled.index.minute / 60.0
        built["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        built["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        built["dow_sin"] = np.sin(2 * np.pi * resampled.index.dayofweek / 7.0)
        built["dow_cos"] = np.cos(2 * np.pi * resampled.index.dayofweek / 7.0)

        # Forward hazard: does gas exceed this device's own high percentile at
        # any point in the next HORIZON steps?  Per-device, because the three
        # nodes sit at materially different baselines.
        threshold = float(np.nanpercentile(resampled["gas"], HAZARD_PCTL))
        gas = resampled["gas"].to_numpy()
        n = len(resampled)
        hazard = np.zeros(n, dtype=int)
        for i in range(n):
            ahead = gas[i + 1:min(n, i + 1 + HORIZON)]
            if ahead.size and ahead.max() >= threshold:
                hazard[i] = 1

        built["hazard"] = hazard
        built["t"] = np.arange(n)
        built["device"] = device
        built["gas_threshold"] = threshold
        # The final HORIZON rows have no future to label against.
        blocks.append(built.iloc[:-HORIZON].dropna())

    if not blocks:
        raise SystemExit("No device had enough contiguous data to build features.")
    return pd.concat(blocks, ignore_index=True)


def evaluate(data: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Temporal hold-out within each device."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = data.sort_values(["device", "t"]).reset_index(drop=True)
    train_idx, test_idx = [], []
    for _, group in data.groupby("device"):
        cut = int(len(group) * 0.8)
        train_idx += list(group.index[:cut])
        test_idx += list(group.index[cut:])

    train, test = data.loc[train_idx], data.loc[test_idx]
    X_train, X_test = train[feature_cols].to_numpy(), test[feature_cols].to_numpy()
    y_train, y_test = train["hazard"].to_numpy(), test["hazard"].to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise SystemExit("Hazard label is degenerate; adjust HAZARD_PCTL or HORIZON.")

    model = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42),
        method="isotonic", cv=3)
    model.fit(X_train, y_train)
    report = HPO.classifier_report(model, X_test, y_test)

    # Reference 1: the current gas reading used directly as a score.
    persistence_auc = float(roc_auc_score(y_test, test["gas"].to_numpy()))

    # Reference 2: a linear model on identical features.
    linear = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, C=1.0))
    linear.fit(X_train, y_train)
    linear_report = HPO.classifier_report(linear, X_test, y_test)

    return {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "hazard_rate_train": round(float(y_train.mean()), 4),
        "hazard_rate_test": round(float(y_test.mean()), 4),
        "calibrated_gradient_boosting": report,
        "logistic_regression": linear_report,
        "persistence_baseline": {"roc_auc": round(persistence_auc, 4),
                                 "description": "current gas reading used directly as a score"},
        "comparison_metric": "ROC AUC and Brier score for forward gas-hazard within "
                             f"{HORIZON} minutes, against a per-device {HAZARD_PCTL}th "
                             "percentile threshold",
    }


def leave_one_device_out(data: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    Train on two devices, test on the third.

    This is the harder and more honest question: does the pipeline transfer to a
    node it has never seen, which is what happens every time a new bin is
    installed. Within-device temporal splits cannot answer it.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    results = {}
    for device in data["device"].unique():
        train = data[data["device"] != device]
        test = data[data["device"] == device]
        y_train, y_test = train["hazard"].to_numpy(), test["hazard"].to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results[str(device)] = {"skipped": "degenerate label"}
            continue
        model = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                           random_state=42),
            method="isotonic", cv=3)
        model.fit(train[feature_cols].to_numpy(), y_train)
        results[str(device)] = {
            "n_train": int(len(train)), "n_test": int(len(test)),
            "hazard_rate_test": round(float(y_test.mean()), 4),
            **HPO.classifier_report(model, test[feature_cols].to_numpy(), y_test),
        }
    aucs = [r["roc_auc"] for r in results.values() if "roc_auc" in r]
    return {
        "per_held_out_device": results,
        "mean_roc_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "min_roc_auc": round(float(np.min(aucs)), 4) if aucs else None,
        "interpretation": "transfer to a device never seen in training, the analogue of "
                          "commissioning a newly installed bin",
    }


def main() -> None:
    path = resolve_csv(sys.argv)
    print(f"Loading {path} …")
    frame = load(path)
    source = describe_source(frame, path)

    print(f"  {source['raw_rows']:,} rows · {source['devices']} devices · "
          f"{source['duration_days']} days · "
          f"{source['median_sampling_interval_s']}s median interval")

    data = build_features(frame)
    feature_cols = [c for c in data.columns
                    if c not in ("hazard", "t", "device", "gas_threshold")]
    print(f"  {len(data):,} resampled rows · {len(feature_cols)} features · "
          f"hazard rate {data['hazard'].mean():.3f}")

    if len(data) < MIN_ROWS:
        raise SystemExit(f"Only {len(data)} usable rows; need at least {MIN_ROWS}.")

    within = evaluate(data, feature_cols)
    across = leave_one_device_out(data, feature_cols)

    print("\nWithin-device temporal hold-out")
    print(f"  calibrated gradient boosting : AUC {within['calibrated_gradient_boosting']['roc_auc']:.4f}"
          f"  AP {within['calibrated_gradient_boosting']['average_precision']:.4f}"
          f"  Brier {within['calibrated_gradient_boosting']['brier']:.4f}"
          f"  (skill {within['calibrated_gradient_boosting']['brier_skill_score']:.3f})")
    print(f"  logistic regression          : AUC {within['logistic_regression']['roc_auc']:.4f}")
    print(f"  persistence (current gas)    : AUC {within['persistence_baseline']['roc_auc']:.4f}")
    print(f"  test hazard rate             : {within['hazard_rate_test']:.4f}")

    print("\nLeave-one-device-out transfer")
    for device, row in across["per_held_out_device"].items():
        if "roc_auc" in row:
            print(f"  held out {device}: AUC {row['roc_auc']:.4f}  "
                  f"Brier {row['brier']:.4f}  (n={row['n_test']:,})")
    print(f"  mean AUC {across['mean_roc_auc']}  ·  worst {across['min_roc_auc']}")

    payload = {
        "dataset": source,
        "protocol": {
            "resample": RESAMPLE, "rolling_window": ROLL,
            "horizon_steps": HORIZON, "hazard_percentile": HAZARD_PCTL,
            "features": feature_cols,
            "validation": "temporal hold-out within each device (80/20), plus "
                          "leave-one-device-out transfer",
            "scope": "validates forward hazard prediction from noisy real telemetry, not "
                     "bin-fill forecasting: no public dataset carries fill level alongside "
                     "gas, temperature and humidity",
        },
        "within_device": within,
        "leave_one_device_out": across,
    }
    (RESULTS / "realdata.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RESULTS / 'realdata.json'}")


if __name__ == "__main__":
    main()
