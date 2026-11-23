"""
Pothole Reporting Backend
==========================

REST API that receives detection records from onboard vehicle units,
deduplicates reports of the same physical pothole, reverse-geocodes
locations, and serves aggregated data to the municipal dashboard.

Run:
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:5000
"""

import csv
import hmac
import os
import re
import threading
from flask import Flask, request, jsonify

from models import init_db, get_db, insert_or_merge_detection
from geolocation import reverse_geocode

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # The dashboard is served separately (static HTML/JS) and needs
    # cross-origin access to this API. Kept dependency-free (no flask-cors)
    # so the backend has zero extra install requirements beyond Flask itself.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------------------------------------------------------------------------
# Read-only public API
# ---------------------------------------------------------------------------
# Anyone can READ (GET). Changing data (POST / PATCH) is limited to the owner:
#   - If the environment variable ROADSTATE_ADMIN_KEY is set, a write must send
#     the header  X-API-Key: <that value>.
#   - If it is NOT set, writes are only accepted from this same computer
#     (localhost) and never when the request came through a proxy.
# This lets you seed data with the simulator while visitors can only view it.
ADMIN_KEY = os.environ.get("ROADSTATE_ADMIN_KEY")


def writes_allowed() -> bool:
    if ADMIN_KEY:
        return hmac.compare_digest(request.headers.get("X-API-Key", ""), ADMIN_KEY)
    is_local = request.remote_addr in ("127.0.0.1", "::1")
    via_proxy = "X-Forwarded-For" in request.headers
    return is_local and not via_proxy


@app.before_request
def block_public_writes():
    if request.path.startswith("/api/") and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if not writes_allowed():
            return jsonify({"error": "This API is read-only for the public."}), 403


ALLOWED_STATUSES = {"reported", "acknowledged", "scheduled", "repaired"}
ALLOWED_SEVERITIES = {"minor", "moderate", "severe"}


def background_geocode(detection_id, lat, lon):
    """Fetches the address in the background so the API doesn't block."""
    try:
        address = reverse_geocode(lat, lon)
        if address:
            with get_db() as conn:
                conn.execute(
                    "UPDATE detections SET address = ? WHERE id = ?", 
                    (address, detection_id)
                )
                conn.commit()
    except Exception as e:
        # In a production app, log this error
        print(f"Geocoding failed for detection {detection_id}: {e}")


def bbox_filter():
    """
    Optional region filter. If min_lat, max_lat, min_lon and max_lon are all
    given as query params, returns (sql_clause, params) restricting results to
    that bounding box. Raises ValueError if they are partial or non-numeric.
    """
    keys = ("min_lat", "max_lat", "min_lon", "max_lon")
    vals = [request.args.get(k) for k in keys]
    if all(v is None for v in vals):
        return "", []
    if any(v is None for v in vals):
        raise ValueError("min_lat, max_lat, min_lon and max_lon must all be provided together")
    min_lat, max_lat, min_lon, max_lon = (float(v) for v in vals)
    clause = " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
    return clause, [min_lat, max_lat, min_lon, max_lon]


@app.route("/api/detections", methods=["POST"])
def submit_detection():
    """
    Body:
    {
      "vehicle_id": "bus-042",
      "latitude": 8.5241,
      "longitude": 76.9366,
      "severity": "moderate",
      "est_depth_cm": 3.2,
      "confidence": 0.87,
      "image_ref": "s3://bucket/frame123.jpg"   # optional
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    required = ["vehicle_id", "latitude", "longitude", "severity"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    if data["severity"] not in ALLOWED_SEVERITIES:
        return jsonify({"error": f"severity must be one of {sorted(ALLOWED_SEVERITIES)}"}), 400

    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (TypeError, ValueError):
        return jsonify({"error": "latitude/longitude must be numeric"}), 400

    # Insert the detection without blocking for the geocoder
    result = insert_or_merge_detection(
        vehicle_id=data["vehicle_id"],
        lat=lat,
        lon=lon,
        severity=data["severity"],
        est_depth_cm=float(data.get("est_depth_cm", 0)),
        confidence=float(data.get("confidence", 0)),
        image_ref=data.get("image_ref"),
        address=None,  # Will be populated in the background
    )

    # Trigger background thread for geocoding if we have a valid ID
    detection_id = result.get("id")
    if detection_id:
        threading.Thread(
            target=background_geocode, 
            args=(detection_id, lat, lon), 
            daemon=True
        ).start()

    status_code = 200 if result.get("merged") else 201
    return jsonify(result), status_code


@app.route("/api/detections", methods=["GET"])
def list_detections():
    """
    Query params:
      severity=minor|moderate|severe   filter by severity
      status=reported|acknowledged...  filter by status
      limit=100                        max rows (default 200)
      offset=0                         pagination offset
      min_lat, max_lat, min_lon, max_lon   restrict to a region (bounding box)
    """
    severity = request.args.get("severity")
    status = request.args.get("status")

    # Safely parse limit and offset to avoid 500 errors
    try:
        limit = min(int(request.args.get("limit", 200)), 1000)
    except ValueError:
        return jsonify({"error": "limit must be a valid integer"}), 400

    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "offset must be a valid integer"}), 400

    try:
        bbox_sql, bbox_params = bbox_filter()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    query = "SELECT * FROM detections WHERE 1=1" + bbox_sql
    params = list(bbox_params)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if status:
        query += " AND status = ?"
        params.append(status)
        
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return jsonify([dict(r) for r in rows])


@app.route("/api/detections/<detection_id>", methods=["GET"])
def get_detection(detection_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/detections/<detection_id>", methods=["PATCH"])
def update_detection(detection_id):
    """Body: {"status": "acknowledged"} — used by municipal officials via the dashboard."""
    data = request.get_json(force=True, silent=True) or {}
    new_status = data.get("status")
    
    if new_status not in ALLOWED_STATUSES:
        return jsonify({"error": f"status must be one of {sorted(ALLOWED_STATUSES)}"}), 400

    with get_db() as conn:
        cur = conn.execute(
            "UPDATE detections SET status = ? WHERE id = ?", 
            (new_status, detection_id)
        )
        conn.commit()  # Explicitly commit the transaction
        
        if cur.rowcount == 0:
            return jsonify({"error": "not found"}), 404
            
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()

    return jsonify(dict(row))


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        bbox_sql, bbox_params = bbox_filter()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE 1=1" + bbox_sql, bbox_params
        ).fetchone()["c"]
        by_severity = conn.execute(
            "SELECT severity, COUNT(*) c FROM detections WHERE status != 'repaired'"
            + bbox_sql + " GROUP BY severity", bbox_params
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) c FROM detections WHERE 1=1"
            + bbox_sql + " GROUP BY status", bbox_params
        ).fetchall()
        vehicles = conn.execute(
            "SELECT COUNT(DISTINCT vehicle_id) c FROM detections WHERE 1=1" + bbox_sql, bbox_params
        ).fetchone()["c"]

    return jsonify({
        "total_reports": total,
        "active_vehicles": vehicles,
        "open_by_severity": {r["severity"]: r["c"] for r in by_severity},
        "by_status": {r["status"]: r["c"] for r in by_status},
    })


# ---------------------------------------------------------------------------
# State-wise deaths (read from a CSV that sits next to this file)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DEATHS_CSV = os.path.join(BASE_DIR, "india_pothole_deaths_by_state_2020_2024.csv")


def _to_number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _load_tidy(rows, metric):
    """
    Layout used by india_pothole_deaths_by_state_2020_2024.csv:
        region_type, region, metric, period, value, source_note
    One row per region + metric + period. period is a year ("2024") or a
    range ("2020-2024"). Only rows whose metric matches are used.
    """
    cols = {k.strip().lower(): k for k in rows[0]}
    rt, rg, me, pe, va = (cols[k] for k in ("region_type", "region", "metric", "period", "value"))
    src = cols.get("source_note")

    states, national_by_year, sources = {}, {}, []
    file_total, period_label = None, None

    for r in rows:
        if (r[me] or "").strip().lower() != metric.lower():
            continue
        value = _to_number(r[va])
        period = (r[pe] or "").strip()
        rng = re.fullmatch(r"(20\d\d)\s*[-\u2013\u2014]\s*(20\d\d)", period)
        year = period if re.fullmatch(r"20\d\d", period) else None
        if value is None or not (rng or year):
            continue

        name = (r[rg] or "").strip()
        if src and (r[src] or "").strip() and r[src].strip() not in sources:
            sources.append(r[src].strip())

        if (r[rt] or "").strip().lower() == "national" or name.lower() in ("india", "all india"):
            if rng:
                file_total, period_label = value, f"{rng[1]}\u2013{rng[2]}"
            else:
                national_by_year[year] = int(value)
            continue

        entry = states.setdefault(name, {"state": name, "period_total": None, "by_year": {}})
        if rng:
            entry["period_total"] = value
        else:
            entry["by_year"][year] = int(value)

    out = []
    for e in states.values():
        # The 2020-2024 figure is the total; single-year rows are parts of it, so never add them on top.
        deaths = e["period_total"] if e["period_total"] is not None else sum(e["by_year"].values())
        out.append({"state": e["state"], "deaths": int(round(deaths)),
                    "by_year": dict(sorted(e["by_year"].items())),
                    "total_is_partial": e["period_total"] is None})
    out.sort(key=lambda d: -d["deaths"])

    years = sorted(national_by_year)
    return {
        "states": out,
        "national_by_year": national_by_year,
        "total": int(file_total) if file_total else sum(d["deaths"] for d in out),
        "period": period_label or (f"{years[0]}\u2013{years[-1]}" if years else ""),
        "metric": metric,
        "sources": sources,
    }


def load_state_deaths(path=STATE_DEATHS_CSV, metric="pothole_deaths"):
    """
    Reads the state-wise deaths CSV. Accepts three common layouts:
      1. one column per year      State, 2020, 2021, ..., 2024 [, Total]
      2. one total column         State, Deaths (or "Total deaths 2020-2024")
      3. one row per state+year   State, Year, Deaths
    State names are taken from the column whose header contains "state".
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"states": [], "national_by_year": {}, "total": 0, "period": ""}

    header = {(c or "").strip().lower() for c in rows[0]}
    if {"region_type", "region", "metric", "period", "value"} <= header:
        result = _load_tidy(rows, metric)
        result["source_file"] = os.path.basename(path)
        return result

    cols = list(rows[0].keys())
    low = {c: (c or "").strip().lower() for c in cols}
    state_col = next((c for c in cols if "state" in low[c] or low[c] in ("ut", "region")), cols[0])
    others = [c for c in cols if c != state_col]

    def is_total(c):
        return "total" in low[c] or re.search(r"20\d\d\D+20\d\d", low[c]) is not None

    year_of = lambda c: (re.search(r"(20\d\d)", low[c]) or [None, None])[1]
    year_cols = [c for c in others if year_of(c) and not is_total(c)]
    total_col = next((c for c in others if is_total(c)), None)
    long_year_col = next((c for c in others if "year" in low[c] and not year_of(c)), None)

    skip_names = {"total", "india", "all india", "grand total", "national"}
    states, national_by_year, file_total = {}, {}, None

    def add(name, year, value):
        entry = states.setdefault(name, {"state": name, "deaths": 0, "by_year": {}})
        if year:
            entry["by_year"][year] = entry["by_year"].get(year, 0) + value
        entry["deaths"] += value

    if long_year_col:                                           # layout 3
        value_col = next((c for c in others if c != long_year_col and
                          ("death" in low[c] or "fatal" in low[c])), None) \
                    or [c for c in others if c != long_year_col][-1]
        for r in rows:
            name, v = (r[state_col] or "").strip(), _to_number(r[value_col])
            if not name or v is None:
                continue
            if name.lower() in skip_names:
                continue
            add(name, str(r[long_year_col]).strip(), v)
    else:                                                       # layouts 1 and 2
        for r in rows:
            name = (r[state_col] or "").strip()
            if not name:
                continue
            if name.lower() in skip_names:
                if total_col and _to_number(r[total_col]) is not None:
                    file_total = _to_number(r[total_col])
                continue
            if year_cols:
                vals = {year_of(c): _to_number(r[c]) or 0 for c in year_cols}
                for y, v in vals.items():
                    add(name, y, v)
            else:
                col = total_col or (others[0] if others else None)
                v = _to_number(r[col]) if col else None
                if v is not None:
                    add(name, None, v)

    out = sorted(states.values(), key=lambda d: -d["deaths"])
    for d in out:
        d["deaths"] = int(round(d["deaths"]))
        d["by_year"] = {y: int(round(v)) for y, v in sorted(d["by_year"].items())}
        for y, v in d["by_year"].items():
            national_by_year[y] = national_by_year.get(y, 0) + v
    years = sorted(national_by_year)
    return {
        "states": out,
        "national_by_year": national_by_year,
        "total": int(file_total) if file_total else sum(d["deaths"] for d in out),
        "period": f"{years[0]}–{years[-1]}" if years else "",
        "source_file": os.path.basename(path),
    }


@app.route("/api/state-deaths", methods=["GET"])
def state_deaths():
    try:
        return jsonify(load_state_deaths(metric=request.args.get("metric", "pothole_deaths")))
    except FileNotFoundError:
        return jsonify({"error": "CSV not found next to app.py: " + os.path.basename(STATE_DEATHS_CSV)}), 404


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    import os
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)