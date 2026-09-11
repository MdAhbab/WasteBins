"""
The Wyndham open smart-bin dataset as a routing instance.
=========================================================

Wyndham City Council publishes the position and daily fill reading of 33
public-place compacting containers in Werribee and Point Cook, Victoria, under
CC-BY 4.0.  Three properties make it usable here and one makes it limited, and
both halves are stated rather than glossed.

What it gives us.  Real container positions, so the road distances are between
places a vehicle actually has to reach.  Real fill trajectories on 966 observed days,
so the demand pattern is observed rather than generated.  Two waste streams at
the same standing, general and recycling, which turns the stream-licensing
constraint in the planner from a hypothetical into a binding one.

What it does not give us.  Fill is quantised to six levels, sampled daily, with
no collection timestamps, no vehicle traces, no mass and no fault labels.  It
therefore drives the demand side of a routing instance and cannot validate the
forecast, the fault layer or the emissions model.  Those stay on the simulated
Dhaka network, where ground truth exists.

The raw file is 13 MB of GeoJSON.  It is parsed once into two small tables under
``data/wyndham/`` which are committed, so the instance builds offline.

Source: https://data.gov.au/data/dataset/wyndham-smart-bin-fill-level-historical
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "wyndham"
CONTAINERS_CSV = DATA_DIR / "containers.csv"
FILL_CSV = DATA_DIR / "fill_daily.csv"

SOURCE_URL = (
    "https://data.gov.au/data/dataset/660a87c3-480e-498b-bfc3-ec84dc504b1c/"
    "resource/58ef5329-3141-4019-be41-24f2584256bc/download/"
    "wyndham_smartbin_filllevel.json"
)

#: The feed reports fullness on a six-point ordinal scale.  Dividing by the top
#: of the scale is the only defensible mapping to a fraction, and it is coarse:
#: one step is a fifth of the container.  Every result on this network inherits
#: that quantisation and the paper says so.
FULLNESS_MAX = 10.0

#: Nominal capacity of a Bigbelly-class solar compacting container in litres,
#: and the in-container density after compaction.  Both are needed to turn a
#: fill fraction into a mass, and neither is published by the council, so they
#: are stated here as assumptions rather than buried in a call site.
CAPACITY_L = 600.0
DENSITY_KG_PER_M3 = 320.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _stream_of(description: str) -> str:
    """Read the waste stream out of the council's naming convention."""
    tokens = description.split()
    if "R" in tokens:
        return "recycling"
    if "G" in tokens:
        return "general"
    return "general"


def _suburb_of(description: str) -> str:
    if "Point Cook" in description:
        return "Point Cook"
    if "Werribee" in description:
        return "Werribee"
    return "unknown"


def parse(raw_path: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the raw GeoJSON into a container table and a daily fill table."""
    payload = json.loads(pathlib.Path(raw_path).read_text(encoding="utf-8"))
    rows: List[dict] = []
    for feature in payload["features"]:
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"]
        rows.append(dict(
            serial=int(props["serialNumber"]),
            date=props["timestamp"],
            fullness=float(props["latestFullness"]),
            reason=props["reason"],
            threshold=float(props["fullnessThreshold"]),
            description=props["description"],
            latitude=float(lat),
            longitude=float(lon),
        ))
    frame = pd.DataFrame(rows)

    containers = (frame.groupby("serial")
                  .agg(description=("description", "first"),
                       latitude=("latitude", "median"),
                       longitude=("longitude", "median"),
                       threshold=("threshold", "first"),
                       n_readings=("fullness", "size"))
                  .reset_index())
    containers["stream"] = containers["description"].map(_stream_of)
    containers["suburb"] = containers["description"].map(_suburb_of)

    fill = frame[["serial", "date", "fullness", "reason"]].copy()
    fill = fill.sort_values(["date", "serial"]).reset_index(drop=True)
    return containers, fill


def download(force: bool = False) -> pathlib.Path:
    """Fetch the raw GeoJSON and write the two cached tables."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONTAINERS_CSV.exists() and FILL_CSV.exists() and not force:
        return CONTAINERS_CSV
    raw = DATA_DIR / "wyndham_smartbin_filllevel.json"
    if not raw.exists() or force:
        req = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "wastebins-research/1.0 (academic)"})
        with urllib.request.urlopen(req, timeout=300) as fh:
            raw.write_bytes(fh.read())
    containers, fill = parse(raw)
    containers.to_csv(CONTAINERS_CSV, index=False)
    fill.to_csv(FILL_CSV, index=False)
    raw.unlink()          # the two tables carry everything we use
    return CONTAINERS_CSV


_LOADED: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None


def load() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the cached tables, downloading once if they are absent.

    The result is held in memory.  A routing study builds one instance per
    observed day and each call would otherwise re-read 31,427 rows and pivot
    them, which dominated the run time of the transfer study and changed no
    result.  Callers receive copies, so a caller that mutates its frame cannot
    corrupt the next one.
    """
    global _LOADED
    if _LOADED is None:
        if not (CONTAINERS_CSV.exists() and FILL_CSV.exists()):
            download()
        _LOADED = (pd.read_csv(CONTAINERS_CSV), pd.read_csv(FILL_CSV))
    return _LOADED[0].copy(), _LOADED[1].copy()


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------
def depot(containers: pd.DataFrame) -> Tuple[float, float]:
    """
    Depot position for the Wyndham instances.

    The council does not publish the standing of the collection vehicle, so we
    place the depot at the centroid of the container set rather than invent an
    address.  This is favourable to every policy equally, and it understates the
    line-haul component that a real out-of-town depot would add.
    """
    return (float(containers["latitude"].mean()),
            float(containers["longitude"].mean()))


def fill_matrix(containers: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    """Dates as rows, container serials as columns, fill fraction in [0, 1]."""
    wide = fill.pivot_table(index="date", columns="serial",
                            values="fullness", aggfunc="last")
    wide = wide.reindex(columns=containers["serial"].tolist())
    return (wide / FULLNESS_MAX).clip(0.0, 1.0)


_SNAPSHOT_CACHE: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}


def snapshots(n_snapshots: int = 25, seed: int = 42,
              min_active: int = 20) -> List[Dict[str, object]]:
    """
    Draw routing snapshots from observed days.

    A day is eligible when at least ``min_active`` containers reported, which
    excludes the commissioning period and the outages.  Days are then sampled
    without replacement at a fixed seed, so the set is reproducible and is not
    the easiest or the busiest days chosen after the fact.
    """
    key = (int(n_snapshots), int(seed), int(min_active))
    if key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[key]
    containers, fill = load()
    wide = fill_matrix(containers, fill)
    eligible = wide.index[wide.notna().sum(axis=1) >= min_active]
    if len(eligible) < n_snapshots:
        raise RuntimeError(
            f"only {len(eligible)} eligible days for {n_snapshots} snapshots")
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(np.asarray(eligible), size=n_snapshots,
                               replace=False).tolist())

    out: List[Dict[str, object]] = []
    for day in picked:
        row = wide.loc[day]
        fills = {int(s): float(v) for s, v in row.items() if np.isfinite(v)}
        out.append(dict(date=str(day), fills=fills))
    _SNAPSHOT_CACHE[key] = out
    return out


def describe() -> Dict[str, object]:
    """Summary statistics quoted in the paper, derived from the cached tables."""
    containers, fill = load()
    wide = fill_matrix(containers, fill)
    active = wide.notna().sum(axis=1)
    return {
        "containers": int(len(containers)),
        "general": int((containers["stream"] == "general").sum()),
        "recycling": int((containers["stream"] == "recycling").sum()),
        "suburbs": containers["suburb"].value_counts().to_dict(),
        "readings": int(len(fill)),
        "days": int(len(wide)),
        "date_first": str(wide.index.min()),
        "date_last": str(wide.index.max()),
        "distinct_levels": int(fill["fullness"].nunique()),
        "mean_fill": float(np.nanmean(wide.to_numpy())),
        "days_with_20_plus_active": int((active >= 20).sum()),
        "alert_share": float((fill["reason"] == "ALERT").mean()),
        "fullness_trigger_share": float((fill["reason"] == "FULLNESS").mean()),
    }


if __name__ == "__main__":
    download()
    print(json.dumps(describe(), indent=2, default=str))
