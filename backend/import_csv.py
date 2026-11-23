"""
Load pothole locations from a CSV into the database, so they appear on the dashboard.

Run from the backend/ folder (no need to start the server first):
    python import_csv.py                          # reads sample_potholes_india.csv
    python import_csv.py --file my_data.csv
    python import_csv.py --file my_data.csv --label "BBMP complaints"

Column names are matched flexibly (case does not matter). Required: a latitude
and a longitude column. Optional: severity, depth, confidence, address, city,
state, vehicle/source. Rows without valid coordinates inside India are skipped
and reported, never silently dropped.
"""

import argparse
import csv
import os
import re

from models import init_db, insert_or_merge_detection

ALIASES = {
    "lat":        ["latitude", "lat", "y"],
    "lon":        ["longitude", "lon", "lng", "long", "x"],
    "severity":   ["severity", "level", "size", "category", "priority"],
    "depth":      ["est_depth_cm", "depth_cm", "depth", "depth_in_cm"],
    "confidence": ["confidence", "conf", "score"],
    "address":    ["address", "location", "place", "road", "street", "area", "locality"],
    "city":       ["city", "district", "town"],
    "state":      ["state", "state_ut", "ut"],
    "source":     ["vehicle_id", "vehicle", "source", "reporter", "reported_by"],
}
SEVERITY_WORDS = {
    "minor": "minor", "low": "minor", "small": "minor", "light": "minor",
    "moderate": "moderate", "medium": "moderate", "mid": "moderate",
    "severe": "severe", "high": "severe", "large": "severe", "critical": "severe", "major": "severe",
}
INDIA_BOUNDS = (6.0, 36.0, 68.0, 98.0)   # south, north, west, east


def pick_columns(header):
    norm = {re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_"): h for h in header if h}
    found = {}
    for field, names in ALIASES.items():
        for n in names:
            if n in norm:
                found[field] = norm[n]
                break
    return found


def number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def severity_from(row, cols):
    if "severity" in cols:
        word = SEVERITY_WORDS.get(str(row[cols["severity"]]).strip().lower())
        if word:
            return word
    depth = number(row.get(cols.get("depth", ""), None))
    if depth is not None:                      # simple fallback from depth in cm
        return "minor" if depth < 3 else "moderate" if depth < 7 else "severe"
    return "moderate"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Import pothole locations from a CSV.")
    ap.add_argument("--file", default=os.path.join(here, "sample_potholes_india.csv"))
    ap.add_argument("--label", default="csv-import", help="Stored as the report's source name")
    args = ap.parse_args()

    init_db()
    with open(args.file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = pick_columns(reader.fieldnames or [])
        print("File:", args.file)
        print("Columns found:", {k: v for k, v in cols.items()})
        if "lat" not in cols or "lon" not in cols:
            raise SystemExit(f"\nCould not find latitude/longitude columns in: {reader.fieldnames}\n"
                             "Rename them to 'latitude' and 'longitude' and try again.")
        if "severity" not in cols and "depth" not in cols:
            print("Note: no severity or depth column, so every row is imported as 'moderate'.")

        s, n, w, e = INDIA_BOUNDS
        new = merged = 0
        skipped = []
        for i, row in enumerate(reader, start=2):          # line 1 is the header
            lat, lon = number(row[cols["lat"]]), number(row[cols["lon"]])
            if lat is None or lon is None:
                skipped.append((i, "missing or non-numeric coordinates"))
                continue
            if not (s <= lat <= n and w <= lon <= e):
                skipped.append((i, f"outside India ({lat}, {lon})"))
                continue

            parts = [str(row[cols[k]]).strip() for k in ("address", "city", "state")
                     if k in cols and str(row[cols[k]]).strip()]
            address = ", ".join(dict.fromkeys(parts)) or None
            conf = number(row.get(cols.get("confidence", ""), None))
            depth = number(row.get(cols.get("depth", ""), None)) or 0.0
            source = str(row.get(cols.get("source", ""), "") or "").strip() or args.label

            result = insert_or_merge_detection(
                vehicle_id=source, lat=lat, lon=lon, severity=severity_from(row, cols),
                est_depth_cm=depth, confidence=conf if conf is not None else 1.0,
                image_ref=None, address=address,
            )
            merged += result["merged"]
            new += not result["merged"]

    print(f"\nDone. {new} new reports, {merged} merged into nearby reports, {len(skipped)} skipped.")
    for line, why in skipped[:10]:
        print(f"  skipped line {line}: {why}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()