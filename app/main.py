import io, os, re, json, math
from pathlib import Path
from typing import List, Dict, Any

import cv2
import fitz
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="Rally Route Mapper")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
STAGE_RE = re.compile(r"(SS\d+)\s*&\s*(SS\d+)\s*[—-]\s*([^—\n]+)", re.I)


def render_pdf(data: bytes):
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []

    for i, page in enumerate(doc):
        # Render the original page image for display.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        out = UPLOADS / f"page_{i+1}.png"
        img.save(out)

        # Extract vector drawing paths directly from the PDF.
        vector_paths = []
        for drawing in page.get_drawings():
            color = drawing.get("color")
            fill = drawing.get("fill")
            items = drawing.get("items", [])

            # Keep paths with a clear red or blue stroke.
            route_type = None
            if color:
                r, g, b = color
                if r > 0.65 and g < 0.4 and b < 0.4:
                    route_type = "special_stage"
                elif b > 0.5 and r < 0.4 and g < 0.5:
                    route_type = "road_section"

            if route_type is None:
                continue

            segments = []
            for item in items:
                kind = item[0]

                if kind == "l":
                    p1, p2 = item[1], item[2]
                    segments.append([
                        [float(p1.x), float(p1.y)],
                        [float(p2.x), float(p2.y)]
                    ])
                elif kind == "c":
                    p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                    segments.append([
                        [float(p1.x), float(p1.y)],
                        [float(p2.x), float(p2.y)],
                        [float(p3.x), float(p3.y)],
                        [float(p4.x), float(p4.y)]
                    ])

            if segments:
                vector_paths.append({
                    "type": route_type,
                    "segments": segments,
                    "width": float(drawing.get("width") or 0)
                })

        # Create a preview showing the extracted vector paths.
        preview = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        preview = preview.copy()
        scale = 2

        for path in vector_paths:
            color = (0, 0, 255) if path["type"] == "special_stage" else (255, 0, 0)

            for segment in path["segments"]:
                points = np.array(
                    [[round(x * scale), round(y * scale)] for x, y in segment],
                    dtype=np.int32
                )

                if len(points) >= 2:
                    cv2.polylines(
                        preview,
                        [points],
                        isClosed=False,
                        color=color,
                        thickness=3,
                        lineType=cv2.LINE_AA
                    )

        preview_name = f"vector_preview_page_{i+1}.png"
        preview_path = UPLOADS / preview_name
        cv2.imwrite(str(preview_path), preview)

        text = page.get_text("text")
        pages.append({
            "page": i + 1,
            "image": f"/uploads/{out.name}",
            "text": text,
            "vector_paths": vector_paths,
            "vector_preview": f"/uploads/{preview_name}"
        })

    return pages

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        out = UPLOADS / f"page_{i+1}.png"
        img.save(out)
        text = page.get_text("text")
        pages.append({"page": i+1, "image": f"/uploads/{out.name}", "text": text})
    return pages

def detect_colours(img: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Rally maps commonly use strong red and blue route lines.
    masks = {
        "special_stage": cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255])) |
                         cv2.inRange(hsv, np.array([168, 100, 80]), np.array([179, 255, 255])),
        "road_section": cv2.inRange(hsv, np.array([90, 70, 60]), np.array([135, 255, 255]))
    }
    result = {}
    for name, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 80:
                continue
            ys, xs = np.where(labels == idx)
            # Downsample component pixels for browser-friendly output.
            pts = np.column_stack([xs, ys])
            step = max(1, len(pts)//1500)
            components.append({"area": area, "points": pts[::step].tolist()})
        result[name] = components
    return result

def create_route_overlay(img: np.ndarray, output_path: Path):
    """Highlight detected red special stages and blue road sections."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = (
        cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
        | cv2.inRange(hsv, np.array([168, 100, 80]), np.array([179, 255, 255]))
    )
    blue_mask = cv2.inRange(
        hsv, np.array([90, 70, 60]), np.array([135, 255, 255])
    )

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    overlay = img.copy()
    overlay[red_mask > 0] = (0, 0, 255)
    overlay[blue_mask > 0] = (255, 80, 0)

    cv2.imwrite(str(output_path), overlay)
    return output_path

def extract_stages(text: str):
    stages = []
    lines = [
        re.sub(r"\s+", " ", line).strip(" \t:—-")
        for line in text.splitlines()
    ]

    pair_re = re.compile(
        r"\b(SS\s*\d+)\s*&\s*(SS\s*\d+)\b",
        re.IGNORECASE
    )
    skip_re = re.compile(
        r"Friday|Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|"
        r"\b\d{1,2}:\d{2}\b|\b\d{4}\b|hrs?|Start|Finish|Regroup|"
        r"Passage Control|Overall Route Map|Shakedown|"
        r"Rev|Crown Copyright|Road Section|Special Stage",
        re.IGNORECASE
    )

    seen = set()

    for i, line in enumerate(lines):
        match = pair_re.search(line)
        if not match:
            continue

        pair = [
            re.sub(r"\s+", "", match.group(1)).upper(),
            re.sub(r"\s+", "", match.group(2)).upper()
        ]

        if tuple(pair) in seen:
            continue
        seen.add(tuple(pair))

        venue = ""
        candidates = [line[match.end():]] + lines[i + 1:i + 6]

        for candidate in candidates:
            candidate = candidate.strip(" \t:—-")
            if not candidate or pair_re.search(candidate):
                continue
            if skip_re.search(candidate):
                continue
            if re.search(r"[A-Za-z]", candidate) and len(candidate) <= 65:
                venue = candidate
                break

        stages.append({
            "stages": pair,
            "name": venue or "Venue not extracted"
        })

    return stages
@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/config")
def config():
    return {"google_maps_configured": bool(os.getenv("GOOGLE_MAPS_API_KEY"))}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf",".png",".jpg",".jpeg",".webp"}:
        raise HTTPException(400, "Upload a PDF or image")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 30 MB")

    if ext == ".pdf":
        pages = render_pdf(data)
    else:
        name = "page_1.png"
        Image.open(io.BytesIO(data)).convert("RGB").save(UPLOADS / name)
        pages = [{"page":1, "image":f"/uploads/{name}", "text":""}]



    for p in pages:
        image_path = UPLOADS / Path(p["image"]).name
        img = cv2.imread(str(image_path))

        p["routes"] = detect_colours(img) if img is not None else {}
        p["stages"] = extract_stages(p["text"])

        if img is not None:
            overlay_name = f"overlay_page_{p['page']}.png"
            overlay_path = UPLOADS / overlay_name
            create_route_overlay(img, overlay_path)
            p["overlay"] = f"/uploads/{overlay_name}"
        else:
            p["overlay"] = None

    return JSONResponse({"filename": file.filename, "pages": pages})

@app.post("/api/geojson")
async def geojson(payload: Dict[str, Any]):
    # Pixel -> lat/lon affine calibration. Three or more control points are required.
    cps = payload.get("control_points", [])
    layers = payload.get("layers", [])
    if len(cps) < 3:
        raise HTTPException(400, "At least 3 calibration points are required")
    A = np.array([[p["x"], p["y"], 1] for p in cps], dtype=float)
    lat = np.array([p["lat"] for p in cps], dtype=float)
    lon = np.array([p["lon"] for p in cps], dtype=float)
    c_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    c_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)

    def transform(pt):
        x, y = pt
        row = np.array([x,y,1.0])
        return [float(row @ c_lon), float(row @ c_lat)]

    features = []
    for layer in layers:
        coords = [transform(p) for p in layer.get("points", [])]
        if len(coords) >= 2:
            features.append({"type":"Feature",
                             "properties":{"type":layer.get("type","route"),
                                           "name":layer.get("name","")},
                             "geometry":{"type":"LineString","coordinates":coords}})
    return {"type":"FeatureCollection","features":features}
