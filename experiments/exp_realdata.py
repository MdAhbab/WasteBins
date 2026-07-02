"""
Optional external validation on REAL IoT sensor telemetry.
==========================================================
The simulation is our controllable primary evaluation. This script additionally
checks that the *same* forward-looking, rolling-feature, calibrated pipeline works
on genuine (noisy, real-world) IoT sensor streams.

Recommended dataset (real, CC0, ~10 MB, well documented):
  "Environmental Sensor Telemetry Data" by Gary Stafford (Kaggle)
  https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k
  Columns: ts, device, co, humidity, light, lpg, motion, smoke, temp
  -> real temperature / humidity / gas (smoke, CO) streams from 3 devices.

WHY this dataset: no public dataset carries bin *fill* level together with gas,
temperature and humidity. This one provides the real hazard modalities
(gas + temperature + humidity), so it validates the HAZARD-prediction component
of our pipeline on real telemetry (predicting a forward gas/air-quality hazard
from the recent past), which is the part most exposed to real sensor noise.

HOW TO GET THE DATA
-------------------
Option A - Kaggle CLI:
    pip install kaggle
    # put your kaggle.json token in %USERPROFILE%\\.kaggle\\ (Account -> Create New API Token)
    kaggle datasets download -d garystafford/environmental-sensor-data-132k \\
        -p experiments/realdata --unzip
Option B - manual:
    Download iot_telemetry_data.csv from the Kaggle page above and place it at
    experiments/realdata/iot_telemetry_data.csv

Then run:  python exp_realdata.py
Results -> results/realdata.json and results/figures/fig_realdata.png

The loader auto-detects column names, so other fill/gas time-series CSVs also work
if you set CSV_PATH and the COLS mapping below.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)

# CSV path resolution order: CLI arg  ->  $REALDATA_CSV  ->  experiments/realdata/  ->  project root
def _resolve_csv():
    import os
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".csv"):
        return pathlib.Path(sys.argv[1])
    if os.environ.get("REALDATA_CSV"):
        return pathlib.Path(os.environ["REALDATA_CSV"])
    local = HERE / "realdata" / "iot_telemetry_data.csv"
    if local.exists():
        return local
    root = HERE.parent.parent.parent / "iot_telemetry_data.csv"   # Micro Paper/ root
    return root if root.exists() else local


CSV_PATH = _resolve_csv()
# column mapping (edit if you use a different dataset)
COLS = {"time": ["ts", "timestamp", "time", "datetime"],
        "group": ["device", "bin_id", "device_id", "id", "node"],
        "temp": ["temp", "temperature"],
        "humidity": ["humidity", "hum"],
        "gas": ["smoke", "gas", "co", "gas_level", "mq135"]}

RESAMPLE = "1min"
ROLL = 10                 # rolling window (steps)
HORIZON = 30              # forward horizon (steps) for the hazard label
HAZ_PCTL = 95            # per-group percentile defining a gas "hazard"


def _pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def load():
    if not CSV_PATH.exists():
        print(f"[!] Dataset not found at {CSV_PATH}\n"
              f"    See the header of this file for one-line download instructions.")
        sys.exit(2)
    df = pd.read_csv(CSV_PATH)
    c = {k: _pick(df, v) for k, v in COLS.items()}
    missing = [k for k, v in c.items() if v is None and k in ("time", "gas")]
    if missing:
        raise SystemExit(f"Could not find columns for {missing}; edit COLS mapping.")
    # timestamp: unix seconds or parseable string
    ts = df[c["time"]]
    dt = pd.to_datetime(ts, unit="s", errors="coerce") if np.issubdtype(ts.dtype, np.number) \
        else pd.to_datetime(ts, errors="coerce")
    df["_dt"] = dt
    df = df.dropna(subset=["_dt"])
    group_col = c["group"] or "_all"
    if group_col == "_all":
        df["_all"] = 0
    return df, c, group_col


def build(df, c, group_col):
    frames = []
    for gid, g in df.groupby(group_col):
        g = g.set_index("_dt").sort_index()
        num = {}
        for key in ("temp", "humidity", "gas"):
            if c[key]:
                num[key] = pd.to_numeric(g[c[key]], errors="coerce")
        r = pd.DataFrame(num).resample(RESAMPLE).mean().interpolate().dropna()
        if len(r) < (ROLL + HORIZON + 20):
            continue
        feats = {}
        for key in num:
            s = r[key]
            feats[key] = s
            feats[f"mean_{key}"] = s.rolling(ROLL, min_periods=1).mean()
            feats[f"std_{key}"] = s.rolling(ROLL, min_periods=2).std().fillna(0.0)
            feats[f"trend_{key}"] = (s - s.shift(ROLL)).fillna(0.0) / ROLL
        fdf = pd.DataFrame(feats)
        fdf["hour"] = r.index.hour
        fdf["dow"] = r.index.dayofweek
        # forward hazard: gas exceeds this group's high percentile within HORIZON
        thr = np.nanpercentile(r["gas"], HAZ_PCTL)
        gas = r["gas"].to_numpy()
        n = len(r)
        haz = np.zeros(n, dtype=int)
        for i in range(n):
            fut = gas[i + 1:min(n, i + 1 + HORIZON)]
            if fut.size and fut.max() >= thr:
                haz[i] = 1
        fdf["hazard"] = haz
        fdf["t"] = np.arange(n)
        fdf["group"] = gid
        frames.append(fdf.dropna())
    return pd.concat(frames, ignore_index=True)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, c, group_col = load()
    data = build(df, c, group_col)
    fcols = [col for col in data.columns if col not in ("hazard", "t", "group")]

    # temporal split within each group (first 80% train / last 20% test)
    data = data.sort_values(["group", "t"]).reset_index(drop=True)
    tr_idx, te_idx = [], []
    for gid, g in data.groupby("group"):
        k = int(len(g) * 0.8)
        tr_idx += list(g.index[:k]); te_idx += list(g.index[k:])
    tr, te = data.loc[tr_idx], data.loc[te_idx]
    Xtr, Xte = tr[fcols].to_numpy(), te[fcols].to_numpy()
    ytr, yte = tr["hazard"].to_numpy(), te["hazard"].to_numpy()

    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42),
        method="isotonic", cv=3)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    gas_col = "gas"
    naive = te[gas_col].to_numpy()   # current gas as a naive predictor

    out = {"dataset": str(CSV_PATH.name), "n_records": int(len(data)),
           "groups": int(data["group"].nunique()),
           "resample": RESAMPLE, "horizon_steps": HORIZON, "hazard_pctl": HAZ_PCTL,
           "hazard_rate_test": float(yte.mean()),
           "calibrated_hgb": {"roc_auc": round(float(roc_auc_score(yte, p)), 4),
                              "avg_precision": round(float(average_precision_score(yte, p)), 4),
                              "brier": round(float(brier_score_loss(yte, p)), 4)},
           "baseline_current_gas": {"roc_auc": round(float(roc_auc_score(yte, naive)), 4)},
           "features": fcols}
    (RES / "realdata.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    ax.plot(mean, frac, "o-", color="#2c7fb8", label="Calibrated HGB (real data)")
    ax.set_xlabel("Mean predicted hazard probability")
    ax.set_ylabel("Observed hazard frequency")
    ax.set_title(f"Real IoT telemetry: forward gas-hazard (AUC={out['calibrated_hgb']['roc_auc']})")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig_realdata.png", dpi=600)
    print("\nSaved results/realdata.json and results/figures/fig_realdata.png")
    print("Paste the AUC/AP/Brier into the commented block in Manuscript.tex / main.tex.")


if __name__ == "__main__":
    main()
