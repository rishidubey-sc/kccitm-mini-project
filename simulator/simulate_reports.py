"""
Fleet simulator for demoing the pothole reporting pipeline without real
hardware. Generates plausible detections from a small fleet of virtual buses
and cars driving fixed routes around a city, and POSTs them to the backend
API — exactly as a real onboard unit would.

Run (with backend already running on :5000):
    python simulate_reports.py --count 40 --interval 0.3 --region greater-noida
"""

import argparse
import os
import random
import time

import requests

API_URL = "http://localhost:5000/api/detections"

# Regions you can simulate (name -> lat_min, lat_max, lon_min, lon_max).
# Keep these in sync with REGIONS in dashboard/dashboard.html.
REGIONS = {
    "delhi-ncr":          (28.20, 28.90, 76.80, 77.70),
    "greater-noida":      (28.40, 28.55, 77.40, 77.60),
    "mumbai":             (18.89, 19.30, 72.77, 73.00),
    "bengaluru":          (12.83, 13.14, 77.46, 77.78),
    "chennai":            (12.90, 13.25, 80.10, 80.35),
    "hyderabad":          (17.28, 17.55, 78.30, 78.60),
    "kolkata":            (22.45, 22.70, 88.25, 88.45),
    "pune":               (18.40, 18.65, 73.75, 73.95),
    "ahmedabad":          (22.95, 23.15, 72.45, 72.70),
    "jaipur":             (26.80, 27.00, 75.70, 75.90),
    "lucknow":            (26.75, 26.95, 80.85, 81.05),
    "thiruvananthapuram": (8.40, 8.60, 76.85, 77.05),
}


def hotspots_for(region_key, n=4):
    """A few fixed 'known pothole' spots per region (same every run), so repeated
    passes produce realistic duplicate-merge behaviour."""
    lat_min, lat_max, lon_min, lon_max = REGIONS[region_key]
    rng = random.Random(region_key)
    return [(rng.uniform(lat_min, lat_max), rng.uniform(lon_min, lon_max)) for _ in range(n)]


HOTSPOTS = {key: hotspots_for(key) for key in REGIONS}


# State / union territory simulation: name -> (anchor towns, spread in degrees).
# Anchors are real towns so simulated potholes land on land, inside the right state.
STATE_ANCHORS = {
    "andhra-pradesh": ([(17.69, 83.22), (16.51, 80.65), (13.63, 79.42)], 0.05),
    "arunachal-pradesh": ([(27.08, 93.61), (28.07, 95.33)], 0.05),
    "assam": ([(26.14, 91.74), (27.48, 94.91), (24.83, 92.8)], 0.05),
    "bihar": ([(25.59, 85.14), (24.8, 85.0), (26.12, 85.39)], 0.05),
    "chhattisgarh": ([(21.25, 81.63), (22.08, 82.14), (21.21, 81.38)], 0.05),
    "goa": ([(15.5, 73.83), (15.27, 73.96)], 0.03),
    "gujarat": ([(23.02, 72.57), (21.17, 72.83), (22.31, 73.18), (22.3, 70.8)], 0.05),
    "haryana": ([(28.46, 77.03), (28.41, 77.31), (30.38, 76.78), (29.15, 75.72)], 0.05),
    "himachal-pradesh": ([(31.1, 77.17), (32.22, 76.32), (31.71, 76.93)], 0.05),
    "jharkhand": ([(23.34, 85.31), (22.8, 86.2), (23.8, 86.43)], 0.05),
    "karnataka": ([(12.97, 77.59), (12.3, 76.64), (15.36, 75.12), (12.91, 74.86)], 0.05),
    "kerala": ([(8.52, 76.94), (9.93, 76.27), (11.26, 75.78)], 0.05),
    "madhya-pradesh": ([(23.26, 77.41), (22.72, 75.86), (23.18, 79.99), (26.22, 78.18)], 0.05),
    "maharashtra": ([(19.08, 72.88), (18.52, 73.86), (21.15, 79.09), (19.99, 73.79)], 0.05),
    "manipur": ([(24.82, 93.94)], 0.05),
    "meghalaya": ([(25.57, 91.88), (25.51, 90.22)], 0.05),
    "mizoram": ([(23.73, 92.72), (22.89, 92.73)], 0.05),
    "nagaland": ([(25.67, 94.11), (25.91, 93.73)], 0.05),
    "odisha": ([(20.3, 85.82), (20.46, 85.88), (22.26, 84.85)], 0.05),
    "punjab": ([(30.9, 75.86), (31.63, 74.87), (31.33, 75.58), (30.34, 76.39)], 0.05),
    "rajasthan": ([(26.91, 75.79), (26.24, 73.02), (24.58, 73.68), (25.18, 75.83)], 0.05),
    "sikkim": ([(27.33, 88.61), (27.17, 88.36)], 0.03),
    "tamil-nadu": ([(13.08, 80.27), (11.02, 76.96), (9.93, 78.12), (10.79, 78.69)], 0.05),
    "telangana": ([(17.39, 78.49), (17.97, 79.59), (18.67, 78.09)], 0.05),
    "tripura": ([(23.83, 91.28), (23.53, 91.49)], 0.03),
    "uttar-pradesh": ([(26.85, 80.95), (28.54, 77.39), (26.45, 80.35), (25.32, 82.97), (27.18, 78.01)], 0.05),
    "uttarakhand": ([(30.32, 78.03), (29.95, 78.16), (29.22, 79.51)], 0.05),
    "west-bengal": ([(22.57, 88.36), (26.73, 88.4), (23.55, 87.31)], 0.05),
    "andaman-and-nicobar-islands": ([(11.62, 92.73)], 0.03),
    "chandigarh": ([(30.73, 76.78)], 0.02),
    "dadra-and-nagar-haveli-and-daman-and-diu": ([(20.27, 73.01), (20.4, 72.83)], 0.02),
    "delhi": ([(28.61, 77.21), (28.59, 77.05)], 0.05),
    "jammu-and-kashmir": ([(34.08, 74.8), (32.73, 74.87)], 0.05),
    "ladakh": ([(34.15, 77.58), (34.56, 76.13)], 0.05),
    "lakshadweep": ([(10.57, 72.64)], 0.02),
    "puducherry": ([(11.93, 79.83)], 0.02),
}


def state_hotspots(key, n=4):
    """Fixed 'known pothole' spots per state (same every run) for realistic merging."""
    anchors, spread = STATE_ANCHORS[key]
    rng = random.Random(key)
    out = []
    for i in range(n):
        lat, lon = anchors[i % len(anchors)]
        out.append((lat + rng.uniform(-spread, spread), lon + rng.uniform(-spread, spread)))
    return out


STATE_HOTSPOTS = {key: state_hotspots(key) for key in STATE_ANCHORS}

VEHICLES = [f"bus-{i:03d}" for i in range(1, 6)] + [f"car-{i:03d}" for i in range(100, 106)]

SEVERITIES = ["minor", "moderate", "severe"]
SEVERITY_WEIGHTS = [0.5, 0.35, 0.15]  # most potholes are minor; severe ones are rarer


def jitter(lat, lon, meters=8):
    """Nudge coordinates slightly, simulating GPS noise / slightly different lane position."""
    deg_per_m = 1 / 111_000
    return (
        lat + random.uniform(-meters, meters) * deg_per_m,
        lon + random.uniform(-meters, meters) * deg_per_m,
    )


def random_location(region_key):
    # "india" = spread reports across all 28 states and 8 union territories.
    if region_key == "india":
        region_key = random.choice(list(STATE_ANCHORS))

    if region_key in STATE_ANCHORS:
        anchors, spread = STATE_ANCHORS[region_key]
        if random.random() < 0.6:                       # re-pass over a known pothole
            lat, lon = random.choice(STATE_HOTSPOTS[region_key])
            return jitter(lat, lon, meters=6)
        lat, lon = random.choice(anchors)               # brand-new pothole near a town
        return (lat + random.uniform(-spread, spread), lon + random.uniform(-spread, spread))

    lat_min, lat_max, lon_min, lon_max = REGIONS[region_key]
    if random.random() < 0.6:
        lat, lon = random.choice(HOTSPOTS[region_key])
        return jitter(lat, lon, meters=6)
    return (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))


def generate_report(region_key):
    lat, lon = random_location(region_key)
    severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
    depth_by_severity = {"minor": (0.5, 2.4), "moderate": (2.5, 4.9), "severe": (5.0, 9.0)}
    depth_range = depth_by_severity[severity]

    return {
        "vehicle_id": random.choice(VEHICLES),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "severity": severity,
        "est_depth_cm": round(random.uniform(*depth_range), 1),
        "confidence": round(random.uniform(0.6, 0.98), 2),
        "image_ref": None,
    }


HEADERS = {}


def run(count: int, interval: float, api_url: str, region: str):
    print(f"Simulating {count} vehicle detections in '{region}' -> {api_url}")
    ok, merged, failed = 0, 0, 0

    for i in range(count):
        report = generate_report(region)
        try:
            resp = requests.post(api_url, json=report, headers=HEADERS, timeout=3)
            resp.raise_for_status()
            result = resp.json()
            was_merged = result.get("merged", False)
            merged += int(was_merged)
            ok += 1
            tag = "MERGED " if was_merged else "NEW    "
            print(f"[{i+1}/{count}] {tag} {report['vehicle_id']:>9} "
                  f"{report['severity']:<8} depth={report['est_depth_cm']}cm "
                  f"@ ({report['latitude']}, {report['longitude']})")
        except requests.RequestException as e:
            failed += 1
            print(f"[{i+1}/{count}] FAILED to submit report: {e}")

        time.sleep(interval)

    print(f"\nDone. {ok} submitted ({merged} merged into existing reports), {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a fleet of vehicles reporting potholes.")
    parser.add_argument("--count", type=int, default=40, help="Number of detections to simulate")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between submissions")
    parser.add_argument("--api-url", default=API_URL, help="Backend ingestion endpoint")
    parser.add_argument("--region", default="india", choices=["india"] + sorted(REGIONS) + sorted(STATE_ANCHORS),
                        help="Where to place the simulated potholes ('india' = every state and union territory)")
    parser.add_argument("--api-key", default=os.environ.get("ROADSTATE_ADMIN_KEY"),
                        help="Admin key, only needed if the backend sets ROADSTATE_ADMIN_KEY")
    args = parser.parse_args()
    if args.api_key:
        HEADERS["X-API-Key"] = args.api_key

    run(args.count, args.interval, args.api_url, args.region)