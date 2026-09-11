"""
Compile the study-area street graphs once and cache them.
=========================================================

This is the only script in the project that queries OpenStreetMap.  It is run by
hand when a study area is added or refreshed, and it writes a compact ``.npz``
under ``data/roadnet/`` that is committed to the repository.  Everything else
reads that cache, so a clean checkout reproduces every routing number without
network access and without depending on the state of the OSM database on the day
of the re-run.

Two study areas.

``dhaka``
    Mirpur, Dhaka.  The 30-container network of ``experiments/sim.py`` is placed
    on real Mirpur coordinates, and this is the bounding box that contains them
    with room for the jitter applied to the synthetic nodes.

``wyndham``
    Werribee and Point Cook, Victoria.  This is the study area of the Wyndham
    City Council open smart-bin dataset, which publishes the real position of 33
    public-place containers together with three years of daily fill readings.
    It is the transfer network: real container positions on a real street graph.

Run:  python -m experiments.build_roadnets [--force]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import roadnet as RN            # noqa: E402

AREAS = {
    # name: (south, west, north, east), region speed profile
    "dhaka": ((23.755, 90.330, 23.850, 90.410), "dhaka"),
    "wyndham": ((-37.925, 144.640, -37.865, 144.755), "wyndham"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="refetch even if a cached graph exists")
    ap.add_argument("--only", default=None, help="build one area only")
    args = ap.parse_args()

    for name, (bbox, region) in AREAS.items():
        if args.only and name != args.only:
            continue
        path = RN.cache_path(name)
        if path.exists() and not args.force:
            net = RN.load(name)
            print(f"[cached] {name}: {net.n_nodes} nodes, {net.n_arcs} arcs, "
                  f"fingerprint {RN.fingerprint(net)}")
            continue
        print(f"[fetch ] {name} {bbox} ...", flush=True)
        net = RN.build(name, bbox, region)
        print(f"[built ] {name}: {net.n_nodes} nodes, {net.n_arcs} arcs, "
              f"fingerprint {RN.fingerprint(net)}")
        print(json.dumps(net.meta, indent=2)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
