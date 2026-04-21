"""
WasteBins – Dummy Data Sender
==============================
Sends simulated sensor readings for 6 bins every 10 seconds.
Uses the authenticated DRF v1 API exclusively.

Flow:
  1. Login via POST /api/v1/auth/login/   → Django session cookie
  2. Bootstrap: POST /api/v1/nodes/ensure/ for each bin → get real DB ids
  3. Loop every INTERVAL seconds:
       POST /api/v1/readings/submit/ for each bin
       POST /api/compute-route/             → triggers route computation
  4. Ctrl-C to stop

Usage:
    python send_dummy_data.py
    python send_dummy_data.py --username admin --password ahbab123 --interval 10
"""
import argparse
import random
import time
from datetime import datetime

try:
    import requests
except ImportError:
    raise SystemExit("Run:  pip install requests")

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_URL      = "http://localhost:8000"
DEFAULT_INTERVAL = 10        # seconds between cycles
USER_LAT         = 23.8103   # Mirpur 10 start point
USER_LNG         = 90.3644

# 6 bins in Mirpur
BINS = [
    {"name": "Bin-A (Mirpur 10)",      "lat": 23.8103, "lng": 90.3644, "fill": 0.30, "id": None},
    {"name": "Bin-B (Sony Square)",    "lat": 23.7910, "lng": 90.3550, "fill": 0.72, "id": None},
    {"name": "Bin-C (Mirpur Stadium)", "lat": 23.8050, "lng": 90.3630, "fill": 0.55, "id": None},
    {"name": "Bin-D (Mirpur 11)",      "lat": 23.8203, "lng": 90.3650, "fill": 0.88, "id": None},
    {"name": "Bin-E (Mirpur 14)",      "lat": 23.8100, "lng": 90.3780, "fill": 0.40, "id": None},
    {"name": "Bin-F (Pallabi)",        "lat": 23.8340, "lng": 90.3650, "fill": 0.65, "id": None},
]

# Fill drift per bin per cycle (± small random noise added each time)
DRIFT = [0.05, -0.02, 0.07, -0.09, 0.06, 0.04]
# ──────────────────────────────────────────────────────────────────────────────


def build_headers(csrf: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-CSRFToken": csrf,
        "Referer": DEFAULT_URL,
    }


def get_csrf(session: requests.Session, base: str) -> str:
    """Fetch a fresh CSRF token from the legacy /api/csrf/ endpoint."""
    r = session.get(f"{base}/api/csrf/", timeout=5)
    r.raise_for_status()
    # The endpoint may return JSON or set a cookie; try both
    try:
        token = r.json().get("csrfToken", "")
    except Exception:
        token = ""
    if not token:
        token = session.cookies.get("csrftoken", "")
    return token


def login(session: requests.Session, base: str, username: str, password: str) -> str:
    """POST credentials to DRF auth endpoint, return fresh CSRF token."""
    csrf = get_csrf(session, base)
    r = session.post(
        f"{base}/api/v1/auth/login/",
        json={"username": username, "password": password},
        headers=build_headers(csrf),
        timeout=5,
    )
    if not r.ok:
        raise SystemExit(f"Login failed [{r.status_code}]: {r.text[:200]}")
    print(f"  Logged in as '{r.json().get('username')}'")
    return get_csrf(session, base)   # refresh token post-login


def bootstrap_nodes(session: requests.Session, base: str, csrf: str) -> str:
    """
    Call /api/v1/nodes/ensure/ for each bin to get (or create) their DB IDs.
    Returns a refreshed CSRF token after all POSTs.
    """
    print("\nBootstrapping nodes …")
    for b in BINS:
        r = session.post(
            f"{base}/api/v1/nodes/ensure/",
            json={"name": b["name"], "latitude": b["lat"], "longitude": b["lng"]},
            headers=build_headers(csrf),
            timeout=5,
        )
        if r.ok:
            data = r.json()
            b["id"] = data["id"]
            action = "CREATED" if data.get("created") else "found  "
            print(f"  [{action}] {b['name']!r:35s}  id={b['id']}")
        else:
            print(f"  [FAILED] {b['name']!r}: {r.status_code} {r.text[:100]}")
        csrf = get_csrf(session, base)   # refresh after each mutating request

    found = sum(1 for b in BINS if b["id"] is not None)
    print(f"  {found}/{len(BINS)} nodes ready.\n")
    return csrf


def send_reading(session: requests.Session, base: str, csrf: str, b: dict) -> bool:
    """Submit one sensor reading for bin b. Returns True on success."""
    fill = b["fill"]
    payload = {
        "node_id":    b["id"],
        "waste_level": round(fill, 3),
        "temperature": round(random.uniform(28.0, 40.0), 1),
        "humidity":    round(random.uniform(60.0, 95.0), 1),
        "gas_level":   round(min(1.0, max(0.0, fill * 0.65 + random.uniform(-0.06, 0.18))), 3),
        "traffic_density": round(random.uniform(0.0, 1.0), 3),
    }
    r = session.post(
        f"{base}/api/v1/readings/submit/",
        json=payload,
        headers=build_headers(csrf),
        timeout=5,
    )
    return r.ok


def trigger_route(session: requests.Session, base: str, csrf: str) -> None:
    """Request an optimal route from the legacy compute-route endpoint."""
    r = session.post(
        f"{base}/api/compute-route/",
        json={"user_lat": USER_LAT, "user_lng": USER_LNG, "alpha": 0.5, "top_n": 6},
        headers=build_headers(csrf),
        timeout=10,
    )
    if r.ok:
        d = r.json()
        path = d.get("route", {}).get("path", [])
        cost = d.get("total_cost", 0)
        algo = d.get("algorithm", "?")
        print(f"  Route: path={path}  cost={cost:.1f}  [{algo}]")
    else:
        print(f"  Route compute failed [{r.status_code}]: {r.text[:120]}")


def run_cycle(session: requests.Session, base: str, cycle: int, interval: float) -> None:
    """Main loop: send readings for all bins then compute route."""
    while True:
        csrf = get_csrf(session, base)
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Cycle #{cycle} — sending {len(BINS)} readings …")

        sent = 0
        for i, b in enumerate(BINS):
            if b["id"] is None:
                print(f"  SKIP {b['name']!r} (no id)")
                continue

            ok = send_reading(session, base, csrf, b)
            if ok:
                sent += 1
                pct = b["fill"] * 100
                tag = "CRIT" if b["fill"] >= 0.85 else "WARN" if b["fill"] >= 0.65 else "OK  "
                print(f"  [{tag}] {b['name']:<32} waste={pct:5.1f}%")

            # Drift fill level realistically
            drift = DRIFT[i] + random.uniform(-0.025, 0.025)
            b["fill"] = round(min(1.0, max(0.05, b["fill"] + drift)), 3)
            if b["fill"] >= 0.98:
                b["fill"] = round(random.uniform(0.05, 0.15), 3)
                print(f"  *** {b['name']} collected -> reset to {b['fill']*100:.0f}%")

            csrf = get_csrf(session, base)

        if sent > 0:
            trigger_route(session, base, csrf)

        print(f"  Done. {sent}/{len(BINS)} readings sent.\n")
        time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description="WasteBins dummy data sender (v2)")
    p.add_argument("--url",      default=DEFAULT_URL,      help="Django base URL")
    p.add_argument("--interval", default=DEFAULT_INTERVAL, type=float, help="Seconds between cycles")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    args = p.parse_args()

    base = args.url.rstrip("/")
    session = requests.Session()
    session.headers["Accept"] = "application/json"

    print(f"\n=== WasteBins Dummy Data Sender ===")
    print(f"Target: {base}\n")

    # ── Credentials ──────────────────────────────────────────────────────────
    username = args.username or input("Username (or email): ").strip()
    password = args.password or input("Password: ").strip()

    csrf = login(session, base, username, password)
    csrf = bootstrap_nodes(session, base, csrf)

    ready = [b for b in BINS if b["id"] is not None]
    if not ready:
        raise SystemExit("No nodes were created/found. Check Django logs.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    cycle = 1
    try:
        run_cycle(session, base, cycle, args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user. Goodbye!")


if __name__ == "__main__":
    main()
