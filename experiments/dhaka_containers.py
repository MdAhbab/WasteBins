"""
Real container positions for Dhaka, from OpenStreetMap.

The earlier version of this study placed containers uniformly at random inside a
bounding box. That is a defensible way to generate an instance and it is a weak
answer to a reviewer who asks whether the geometry is realistic, because random
points do not sit where a city puts its containers: on street frontages, at
market edges, clustered around commercial blocks and absent from open ground.

OpenStreetMap records waste facilities as tagged features, and Dhaka is mapped
well enough to supply several hundred. Using them makes every container position
in this study a place a vehicle actually has to reach, in both cities.

Four tags are read and each is treated differently, because they are different
objects:

``waste_disposal``
    A large container or skip for general refuse. This is the collection
    container the routing problem is about.
``waste_basket``
    A street litter bin. Municipalities collect these too, on a smaller vehicle,
    so they enter the instance with a much smaller capacity.
``recycling``
    A recycling point, entered on the recyclable stream so the vehicle licensing
    constraint binds.
``waste_transfer_station``
    Not a container. These are where a collection vehicle tips, so one of them
    serves as the depot instead of an invented address.

What this does not give us: fill levels. No fill data exists for Dhaka, so the
fill, gas, temperature and humidity of each container are simulated as before.
The positions are real and the demand is not, and the article says so.

Run:  python -m experiments.dhaka_containers
Out:  data/dhaka/containers.csv
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "dhaka"
CONTAINERS_CSV = DATA_DIR / "containers.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Study area: Dhaka city, wide enough to contain the mapped waste facilities
#: and matching the road graph built for the same box.
BBOX = (23.72, 90.33, 23.90, 90.45)

AMENITIES = ("waste_disposal", "waste_basket", "recycling",
             "waste_transfer_station")

#: Nominal capacity in litres and waste stream by tag. A street litter bin is
#: not a 1100 litre container and pretending otherwise would make the capacity
#: constraint meaningless.
PROFILE = {
    "waste_disposal": {"capacity_l": 1100.0, "stream": "general"},
    "waste_basket": {"capacity_l": 240.0, "stream": "general"},
    "recycling": {"capacity_l": 1100.0, "stream": "recyclable"},
}


def _query() -> str:
    south, west, north, east = BBOX
    pattern = "|".join(AMENITIES)
    return (
        "[out:json][timeout:180];"
        f'(node({south},{west},{north},{east})["amenity"~"^({pattern})$"];'
        f' way({south},{west},{north},{east})["amenity"~"^({pattern})$"];);'
        "out center tags;"
    )


def download(force: bool = False) -> pathlib.Path:
    """Fetch the mapped waste facilities and cache them as a table."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONTAINERS_CSV.exists() and not force:
        return CONTAINERS_CSV

    body = ("data=" + _query()).encode("utf-8")
    request = urllib.request.Request(
        OVERPASS_URL, data=body,
        headers={"User-Agent": "wastebins-research/1.0 (academic)"})
    with urllib.request.urlopen(request, timeout=300) as fh:
        payload = json.loads(fh.read().decode("utf-8"))

    rows: List[dict] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {}) or {}
        amenity = tags.get("amenity")
        if amenity not in AMENITIES:
            continue
        point = element if "lat" in element else element.get("center") or {}
        if not point.get("lat"):
            continue
        rows.append(dict(
            osm_type=element.get("type"),
            osm_id=element.get("id"),
            amenity=amenity,
            latitude=float(point["lat"]),
            longitude=float(point["lon"]),
            name=tags.get("name", ""),
            operator=tags.get("operator", ""),
        ))

    frame = pd.DataFrame(rows).drop_duplicates(subset=["osm_type", "osm_id"])
    frame = frame.sort_values(["amenity", "osm_id"]).reset_index(drop=True)
    frame.to_csv(CONTAINERS_CSV, index=False)
    return CONTAINERS_CSV


_LOADED: Optional[pd.DataFrame] = None


def load() -> pd.DataFrame:
    """The cached facility table, downloading once if it is absent."""
    global _LOADED
    if _LOADED is None:
        if not CONTAINERS_CSV.exists():
            download()
        _LOADED = pd.read_csv(CONTAINERS_CSV).fillna({"name": "", "operator": ""})
    return _LOADED.copy()


def containers() -> pd.DataFrame:
    """Collectable containers, meaning everything that is not a transfer station."""
    frame = load()
    frame = frame[frame["amenity"] != "waste_transfer_station"].copy()
    frame["capacity_l"] = frame["amenity"].map(lambda a: PROFILE[a]["capacity_l"])
    frame["stream"] = frame["amenity"].map(lambda a: PROFILE[a]["stream"])
    return frame.reset_index(drop=True)


def depot() -> Tuple[float, float]:
    """
    Depot position: the transfer station most central to the container set.

    A collection vehicle tips at a transfer station, so using one as the depot is
    the closest thing the map offers to the real standing. Choosing the most
    central of them is favourable to every policy equally and is stated rather
    than tuned.
    """
    frame = load()
    stations = frame[frame["amenity"] == "waste_transfer_station"]
    if stations.empty:
        raise RuntimeError("no waste transfer station in the study area")
    body = containers()
    centre_lat = float(body["latitude"].mean())
    centre_lon = float(body["longitude"].mean())
    distance = ((stations["latitude"] - centre_lat) ** 2
                + (stations["longitude"] - centre_lon) ** 2)
    best = stations.loc[distance.idxmin()]
    return (float(best["latitude"]), float(best["longitude"]))


def describe() -> Dict[str, object]:
    frame = load()
    body = containers()
    return {
        "bbox": list(BBOX),
        "facilities": int(len(frame)),
        "by_amenity": frame["amenity"].value_counts().to_dict(),
        "containers": int(len(body)),
        "by_stream": body["stream"].value_counts().to_dict(),
        "named": int((frame["name"].astype(str).str.len() > 0).sum()),
        "with_operator": int((frame["operator"].astype(str).str.len() > 0).sum()),
        "depot": list(depot()),
        "lat_span_km": round(float(body["latitude"].max() - body["latitude"].min()) * 111.0, 1),
        "lon_span_km": round(float(body["longitude"].max() - body["longitude"].min()) * 102.0, 1),
    }


if __name__ == "__main__":
    download()
    print(json.dumps(describe(), indent=2, default=str))
