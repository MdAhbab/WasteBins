# Data

Everything the routing study reads. Five files, two cities, two licences.

Nothing here is synthetic. Container positions are real in both cities, the
Wyndham fill readings are observed, and the street graphs are compiled from
OpenStreetMap. Where the study needs something these files do not supply, such
as Dhaka demand, it is simulated in code and the paper says so.

| File | What it is | Rows | Source |
|---|---|---|---|
| `dhaka/containers.csv` | Mapped waste facilities in Dhaka | 450 | OpenStreetMap |
| `wyndham/containers.csv` | Smart-bin inventory, Werribee and Point Cook | 33 | Wyndham City Council |
| `wyndham/fill_daily.csv` | Daily fill readings | 31,427 | Wyndham City Council |
| `roadnet/dhaka.npz` | Drivable street graph, Dhaka | 134,437 nodes / 272,725 arcs | OpenStreetMap |
| `roadnet/wyndham.npz` | Drivable street graph, Wyndham | 44,180 nodes / 77,543 arcs | OpenStreetMap |

## `dhaka/containers.csv`

Waste facilities OpenStreetMap records inside the bounding box
`(23.72, 90.33, 23.90, 90.45)`, retrieved through the Overpass API.

`osm_type`, `osm_id` identify the feature upstream, so any row can be checked
against OpenStreetMap directly. `amenity` is the raw tag and decides how the row
enters an instance:

| `amenity` | Count | Enters as |
|---|---|---|
| `waste_disposal` | 215 | 1100 L general-waste container |
| `waste_basket` | 212 | 240 L street litter bin |
| `recycling` | 4 | 1100 L recyclable container |
| `waste_transfer_station` | 19 | depot candidate, never a container |

The 431 non-transfer rows are the container population; the study samples 160 of
them. Treating a litter bin as a refuse skip would make the vehicle capacity
constraint meaningless, which is why the classes are separated rather than
pooled.

This is what volunteers have mapped, not the city's asset register, so it is a
lower bound on what exists and may favour well-surveyed areas. Positions are
real; completeness is not claimed.

## `wyndham/containers.csv`

One row per reporting smart bin. `serial` joins to `fill_daily.csv`.
`threshold` is the council's own collection trigger, `n_readings` the number of
daily observations behind that serial, `stream` is `general` or `recycling`
(16 and 17 respectively), and `suburb` is Werribee or Point Cook.

## `wyndham/fill_daily.csv`

`serial,date,fullness,reason`

`fullness` takes six values only, `0 2 4 6 8 10`, so one step is a fifth of a
container. Treat it as an ordinal level rather than a percentage. The study
reads it as a fraction, `level / 10`, which is the only defensible mapping and
is coarse. Turning that fraction into a mass needs a container volume and a
compacted density, and the council publishes neither: 600 L and 320 kg/m3 are
**our assumptions**, stated in `experiments/wyndham.py`. They are not data.

`reason` records why the reading was transmitted, and the split matters when
drawing instances:

| `reason` | Rows | |
|---|---|---|
| `NOT_READY` | 25,776 | below the collection threshold |
| `FULLNESS` | 4,208 | threshold reached |
| `ALERT` | 1,443 | council alert raised |

The series runs 2018-06-26 to 2021-05-03, 966 days from 33 serials, of which 952
days have enough reporting bins to draw an instance from. Fill is zero in 16,359
of the 31,427 readings, so the network is mostly idle and the instances that
matter are the tail.

What it does not contain, and what the paper therefore does not claim: no
collection timestamps, no vehicle traces, no collected mass, no fault labels.

## `roadnet/*.npz`

A compiled directed street graph, reduced to its largest strongly connected
component so every stop is reachable from every other.

| Array | Meaning |
|---|---|
| `lat`, `lon` | node coordinates, indexed by node id |
| `src`, `dst` | arc endpoints as node ids |
| `length_m` | arc length in metres, great-circle between endpoints |
| `speed_kmh` | free-flow speed, from the `maxspeed` tag or the highway class |
| `meta` | the Overpass query, bounding box, retrieval timestamp and fingerprint |

The graph is directed: a two-way segment contributes two arcs and a one-way
segment one, so the resulting distance matrix is asymmetric. Only 4.2 percent of
Dhaka ways declare a speed against 56.1 percent in Wyndham, so Dhaka free-flow
speeds are largely modelled from the highway class. Distances carry no such
assumption.

Load one with:

```python
from wastebins_core import roadnet
net = roadnet.load("dhaka")
distance_m, freeflow_kmh, snap_m = net.matrices(coords)
```

`snap_m` is how far each coordinate sat from the nearest drivable node: at worst
99 m in Dhaka and 55 m in Wyndham.

## Licences

**The data is not under the repository's code licence.** Two upstream licences
apply and both allow redistribution with attribution.

`wyndham/containers.csv` and `wyndham/fill_daily.csv` derive from the Wyndham
Smart Bin Daily Fill Level dataset published by **Wyndham City Council** on
data.gov.au under **Creative Commons Attribution 2.5 Australia**. Attribute the
council. The files here are reshaped, not altered in substance: readings are
pivoted to one row per bin-day and serials are joined to positions.
<https://data.gov.au/data/dataset/wyndham-smart-bin-fill-level-historical>

`dhaka/containers.csv` and both `roadnet/*.npz` files derive from
**OpenStreetMap**, © OpenStreetMap contributors, under the **Open Database
License (ODbL) 1.0**. The street graphs are derived databases, so ODbL's
share-alike term applies to them: redistribute under ODbL and keep the
attribution. <https://www.openstreetmap.org/copyright>

If you use any of this, cite the upstream sources above rather than only this
repository. They did the collection.

## Rebuilding from source

None of these files is hand-edited, and each can be regenerated:

```bash
python -m experiments.build_roadnets       # both roadnet/*.npz
python -m experiments.dhaka_containers     # dhaka/containers.csv
python -m experiments.wyndham              # both wyndham/*.csv
```

`build_roadnets.py` and `dhaka_containers.py` query the Overpass API live, so a
rebuild will pick up map edits made since 11 September 2026 and will not
reproduce these files byte for byte. The fingerprint stored in each `.npz`
identifies the extract the published results were computed on.
