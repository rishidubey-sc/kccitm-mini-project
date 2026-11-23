"""
Detect potholes in ordinary single photos (e.g. from a phone) - no stereo pair needed.

This is the "crowdsourced mobile imagery" path: a person photographs a road, the
model finds potholes, the photo's GPS location (from its EXIF data, or --lat/--lon)
is attached, and each pothole is sent to the backend so it appears on the dashboard.

Severity here uses pothole SIZE in the photo only (no depth sensor), so it is a
rough estimate. The thresholds are the same area cut-offs used in detect_pothole.py.

Examples:
    python detect_image.py --image road.jpg --weights best.pt --out annotated.jpg
    python detect_image.py --image road.jpg --weights best.pt --post-to-backend
    python detect_image.py --image road.jpg --weights best.pt --lat 28.47 --lon 77.50 --post-to-backend
"""

import argparse
import json
from typing import List, Optional, Tuple

import cv2
from PIL import Image
from ultralytics import YOLO

SEVERITY_COLORS = {"minor": (0, 200, 0), "moderate": (0, 165, 255), "severe": (0, 0, 255)}


def severity_from_area(area_px: int, frame_area_px: int) -> str:
    """Size-only severity (placeholder thresholds - calibrate for real use)."""
    ratio = area_px / max(frame_area_px, 1)
    if ratio >= 0.04:
        return "severe"
    if ratio >= 0.015:
        return "moderate"
    return "minor"


def _to_degrees(dms, ref) -> float:
    deg, minutes, seconds = (float(x) for x in dms)
    value = deg + minutes / 60.0 + seconds / 3600.0
    return -value if ref in ("S", "W") else value


def read_exif_gps(image_path: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) from the photo's EXIF GPS tags, or None if it has none."""
    try:
        exif = Image.open(image_path).getexif()
        gps = exif.get_ifd(0x8825)  # GPSInfo
        if not gps or 2 not in gps or 4 not in gps:
            return None
        lat = _to_degrees(gps[2], gps.get(1, "N"))
        lon = _to_degrees(gps[4], gps.get(3, "E"))
        return lat, lon
    except Exception:
        return None


def detect(image_path: str, weights: str, conf: float) -> List[dict]:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = frame.shape[:2]
    result = YOLO(weights).predict(frame, conf=conf, verbose=False)[0]
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = int((x2 - x1) * (y2 - y1))
        out.append({
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "confidence": round(float(box.conf[0]), 2),
            "area_px": area,
            "severity": severity_from_area(area, w * h),
        })
    return out


def draw(image_path: str, detections: List[dict], out_path: str):
    img = cv2.imread(image_path)
    for d in detections:
        x, y, w, h = d["bbox"]
        color = SEVERITY_COLORS[d["severity"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{d['severity']} {int(d['confidence'] * 100)}%", (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, img)


def post_to_backend(detections, api_url, vehicle_id, lat, lon):
    import requests
    for d in detections:
        payload = {
            "vehicle_id": vehicle_id,
            "latitude": lat,
            "longitude": lon,
            "severity": d["severity"],
            "est_depth_cm": 0,          # unknown from a single photo
            "confidence": d["confidence"],
        }
        try:
            r = requests.post(api_url, json=payload, timeout=5)
            r.raise_for_status()
            res = r.json()
            tag = "MERGED into existing report" if res.get("merged") else "NEW report created"
            print(f"  -> {tag}")
        except Exception as e:
            print(f"  -> FAILED to post: {e}\n     Is the backend running at {api_url}?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect potholes in a single photo.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True, help="Trained pothole weights (.pt)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default=None, help="Save an annotated copy of the photo here")
    ap.add_argument("--post-to-backend", action="store_true")
    ap.add_argument("--api-url", default="http://localhost:5000/api/detections")
    ap.add_argument("--vehicle-id", default="mobile-user")
    ap.add_argument("--lat", type=float, default=None, help="Overrides the photo's GPS")
    ap.add_argument("--lon", type=float, default=None, help="Overrides the photo's GPS")
    args = ap.parse_args()

    dets = detect(args.image, args.weights, args.conf)
    print(json.dumps(dets, indent=2))
    print(f"{len(dets)} pothole(s) found.")

    if args.out:
        draw(args.image, dets, args.out)
        print(f"Annotated image saved to {args.out}")

    if args.post_to_backend and dets:
        lat, lon = args.lat, args.lon
        if lat is None or lon is None:
            gps = read_exif_gps(args.image)
            if gps:
                lat, lon = gps
                print(f"Using GPS from the photo: {lat:.5f}, {lon:.5f}")
            else:
                raise SystemExit("This photo has no GPS data. Pass --lat and --lon.")
        post_to_backend(dets, args.api_url, args.vehicle_id, lat, lon)