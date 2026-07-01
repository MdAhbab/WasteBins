"""
Build and cache the simulated dataset once, reuse across all experiments.
Caches:
  results/latent.pkl   -> {bin_id: latent DataFrame}  (for ablation/routing/equity)
  results/features.parquet -> engineered features + forward labels (for ML)
"""
from __future__ import annotations
import pathlib, pickle
import pandas as pd
import sim

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

N_BINS = 30
DAYS = 180
SEED = 42
P_MISSING = 0.03            # realistic sensor dropout for the main dataset

FEATURE_COLS = [
    "waste", "mean_waste", "std_waste", "trend_waste",
    "gas", "mean_gas", "std_gas", "trend_gas",
    "temp", "mean_temp", "std_temp", "trend_temp",
    "humidity", "mean_humidity", "std_humidity", "trend_humidity",
    "hour", "dow",
]


def build(force: bool = False):
    lat_p = RESULTS / "latent.pkl"
    feat_p = RESULTS / "features.parquet"
    if not force and lat_p.exists() and feat_p.exists():
        with open(lat_p, "rb") as f:
            latent = pickle.load(f)
        data = pd.read_parquet(feat_p)
        return latent, data

    bins_df, latent = sim.simulate(n_bins=N_BINS, days=DAYS, seed=SEED)
    for bid in latent:
        latent[bid]["bin_id"] = bid
    frames = []
    for bid, df in latent.items():
        obs = sim.observe(df, seed=1000 + bid, p_missing=P_MISSING)
        feats = sim.build_features(obs)
        feats["bin_id"] = bid
        frames.append(feats)
    data = pd.concat(frames, ignore_index=True)

    with open(lat_p, "wb") as f:
        pickle.dump({"bins_df": bins_df, "latent": latent}, f)
    data.to_parquet(feat_p)
    return {"bins_df": bins_df, "latent": latent}, data


if __name__ == "__main__":
    payload, data = build(force=True)
    print("bins:", len(payload["bins_df"]))
    print("records:", len(data),
          "| hazard rate:", round(float(data["hazard_within_h"].mean()), 3),
          "| overflow-hour rows:", int((data["time_to_overflow"] == 0).sum()))
