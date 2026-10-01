"""
The instances every revised experiment is built from.
=====================================================

One place decides which containers an experiment sees, so that "tuned on" and
"evaluated on" are statements about named sets rather than about whichever seed
a script happened to pass.

Dhaka has 431 mapped containers.  They are divided as follows.

``S0``
    The primary evaluation sample, 160 containers.  It is the sample the first
    version of the study used, drawn the same way, so its snapshots reproduce
    exactly and the street-network measurements made on it still apply.

``T160`` and ``T60``
    Tuning sets, drawn from the 271 containers outside ``S0``.  ``T60`` is the
    first 60 containers of ``T160``.  The prize weight is examined on ``T60``
    and the baseline solvers are configured on ``T160``, so nothing chosen by
    looking at results was chosen on a container the primary sample contains.

``S1`` to ``S4``
    Four further evaluation samples of 160, drawn from the 371 containers
    outside ``T60``.  They vary the geometry, which the snapshots of one sample
    cannot.  They may share containers with ``T160``, because 431 containers do
    not hold a tuning set and five disjoint samples of 160; no weight was tuned
    on ``T160``, only solver settings.

``R0`` to ``R7``
    Rollout networks of 60 containers, drawn from outside ``T60``.

``P0`` and ``P1``
    Pilot rollout networks of 60 containers, drawn from inside ``T160``.  The
    demand level of the rollout is set on these and on nothing else.

Wyndham has 33 containers and is used whole.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import dhaka_containers as DC    # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import wyndham as WY             # noqa: E402

SEED = EF.SEED

#: The instant every plan is costed at, as in the first version of the study.
#: The congestion surface reads the hour and the weekday from it.
WHEN = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)

_NETWORKS: Dict[str, pd.DataFrame] = {}
_SNAPSHOTS: Dict[Tuple, List[Dict]] = {}


def _frame() -> pd.DataFrame:
    return DC.containers()


def network(name: str) -> pd.DataFrame:
    """The container table for a named Dhaka set; see the module docstring."""
    if name in _NETWORKS:
        return _NETWORKS[name]
    frame = _frame()
    primary = frame.sample(n=160, random_state=SEED)

    def outside(excluded: pd.DataFrame) -> pd.DataFrame:
        return frame[~frame["osm_id"].isin(set(excluded["osm_id"]))]

    if name == "S0":
        chosen = primary
    elif name in ("T160", "T60"):
        tuning = outside(primary).sample(n=160, random_state=4242)
        chosen = tuning if name == "T160" else tuning.iloc[:60]
    elif name.startswith("S") and name[1:].isdigit():
        chosen = outside(network("T60")).sample(n=160, random_state=100 + int(name[1:]))
    elif name.startswith("R") and name[1:].isdigit():
        chosen = outside(network("T60")).sample(n=60, random_state=200 + int(name[1:]))
    elif name.startswith("P") and name[1:].isdigit():
        chosen = network("T160").sample(n=60, random_state=300 + int(name[1:]))
    else:
        raise KeyError(f"unknown network {name!r}")
    _NETWORKS[name] = chosen.reset_index(drop=True)
    return _NETWORKS[name]


def depots() -> List[Tuple[float, float]]:
    """
    Mapped transfer stations, most central first.

    The first is the depot of the main study.  The others are used to measure
    how far the results depend on that choice.
    """
    frame = DC.load()
    stations = frame[frame["amenity"] == "waste_transfer_station"].copy()
    body = _frame()
    centre_lat, centre_lon = float(body["latitude"].mean()), float(body["longitude"].mean())
    stations["d2"] = ((stations["latitude"] - centre_lat) ** 2
                      + (stations["longitude"] - centre_lon) ** 2)
    stations = stations.sort_values(["d2", "osm_id"])
    return [(float(r.latitude), float(r.longitude)) for r in stations.itertuples()]


def wyndham_depots() -> Dict[str, Tuple[float, float]]:
    """
    Three depot positions for Wyndham, whose real depot is not published.

    The centroid of all containers is the one the main study uses.  The other
    two put the depot at the centroid of each cluster, which is where it would
    sit if the vehicle were based in either suburb.
    """
    containers, _ = WY.load()
    out = {"centroid": WY.depot(containers)}
    for suburb in ("Werribee", "Point Cook"):
        part = containers[containers["suburb"] == suburb]
        out[suburb.lower().replace(" ", "_")] = (float(part["latitude"].mean()),
                                                 float(part["longitude"].mean()))
    return out


def _stream_seed(name: str) -> int:
    """A snapshot seed per network; the primary sample keeps the original one."""
    if name == "S0":
        return SEED
    return SEED + 7919 * (sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 10007)


def snapshots(name: str, count: int, n_vehicles: int,
              depot: Optional[Tuple[float, float]] = None) -> List[Dict]:
    """
    ``count`` dispatch snapshots of one Dhaka network.

    Snapshots are independent draws of container state on fixed positions.  They
    are not consecutive instants of one trajectory: each is drawn afresh from
    the generator in `exp_fleet.make_snapshot`, so two snapshots share geometry
    and nothing else.
    """
    key = (name, int(count), int(n_vehicles), depot)
    if key not in _SNAPSHOTS:
        frame = network(name)
        EF.set_study("dhaka", n_bins=len(frame), n_vehicles=n_vehicles)
        rng = np.random.default_rng(_stream_seed(name))
        _SNAPSHOTS[key] = [EF.make_snapshot(rng, i, network=frame, depot=depot)
                           for i in range(count)]
    else:
        EF.set_study("dhaka", n_bins=len(network(name)), n_vehicles=n_vehicles)
    return _SNAPSHOTS[key]


def wyndham_snapshots(count: int, n_vehicles: int = 3,
                      depot: Optional[Tuple[float, float]] = None) -> List[Dict]:
    """``count`` observed days of the Wyndham network."""
    key = ("wyndham", int(count), int(n_vehicles), depot)
    EF.set_study("wyndham", n_vehicles=n_vehicles)
    if key not in _SNAPSHOTS:
        rng = np.random.default_rng(SEED)
        _SNAPSHOTS[key] = [EF.make_snapshot(rng, i, depot=depot) for i in range(count)]
    return _SNAPSHOTS[key]


def describe() -> Dict:
    """Sizes and overlaps of the named sets, for the record kept with the results."""
    def ids(name):
        return set(int(x) for x in network(name)["osm_id"])
    s0, t60, t160 = ids("S0"), ids("T60"), ids("T160")
    out = {
        "containers": int(len(_frame())),
        "S0": len(s0), "T60": len(t60), "T160": len(t160),
        "S0_and_T160": len(s0 & t160),
        "T60_inside_T160": len(t60 & t160),
    }
    for k in range(1, 5):
        sk = ids(f"S{k}")
        out[f"S{k}"] = {"size": len(sk), "shared_with_S0": len(sk & s0),
                        "shared_with_T60": len(sk & t60),
                        "shared_with_T160": len(sk & t160)}
    for k in range(8):
        rk = ids(f"R{k}")
        out[f"R{k}"] = {"size": len(rk), "shared_with_T60": len(rk & t60)}
    for k in range(2):
        pk = ids(f"P{k}")
        out[f"P{k}"] = {"size": len(pk), "inside_T160": len(pk & t160),
                        "shared_with_S0": len(pk & s0)}
    out["depots"] = len(depots())
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
    # The primary sample must be the one the first version of the study routed.
    old = EF.set_study("dhaka") and EF.dhaka_network(0)
    assert list(old["osm_id"]) == list(network("S0")["osm_id"]), "S0 changed"
    print("S0 matches the original primary sample")
