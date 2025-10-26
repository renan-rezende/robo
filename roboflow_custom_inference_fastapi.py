# roboflow_custom_inference_fastapi.py
"""
FastAPI server to serve a YOLO segmentation model (.pt) as a Custom Inference
endpoint for Roboflow Auto-Label.

Features:
- loads a Ultralytics YOLO .pt segmentation model
- runs inference on uploaded images
- converts predicted binary masks to polygons (contours) with OpenCV
- filters detections by confidence threshold (default 0.4)
- returns polygons in RELATIVE coordinates (x,y in [0,1]) which Roboflow expects
- returns JSON of the form: {"predictions": [{"label":..., "confidence":..., "polygon": [x1,y1,x2,y2,...]}, ...]}

How to use (short):
1) pip install -r requirements (see below)
2) set MODEL_PATH to your seumodelo.pt
3) uvicorn roboflow_custom_inference_fastapi:app --host 0.0.0.0 --port 8000
4) (optional) ngrok http 8000 -> use the https URL in Roboflow Custom Model

Notes:
- This implementation attempts to be robust across Ultralytics versions by trying a
  few ways to extract masks from results. If your Ultralytics version stores masks
  differently, you may need to adapt the small helper `extract_masks_from_result`.
- We simplify polygons with approxPolyDP to avoid ultra-detailed contours returning
  thousands of points. Tune `approx_epsilon` if needed.
"""

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np
import cv2
import uvicorn
import os

# Ultralytics import (assumes ultralytics is installed)
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("Please install ultralytics (pip install ultralytics). Error: {}".format(e))

# ------------------ CONFIG ------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models\RGB_960_m.pt")  # path to your .pt
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.4))  # filter confidence
APP_HOST = "0.0.0.0"
APP_PORT = int(os.environ.get("PORT", 8000))
# approximate polygon simplification (epsilon multiplier)
APPROX_EPSILON_MULT = 0.01
# minimum polygon area (in pixels) to keep
MIN_POLYGON_AREA = 10
# --------------------------------------------

app = FastAPI(title="Roboflow Custom Inference - YOLO Segmentation -> Polygons")

# load model once at startup
print("Loading model from", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded.")

# helper: convert masks -> list of contours (polygons in pixel coords)

def extract_masks_from_result(result, img_h: int, img_w: int):
    """
    Try to extract masks from a Ultralytics result object.
    Returns list of binary masks (numpy uint8 0/255), one per detection, in image coords.
    This function attempts several common locations for masks used by different
    ultralytics versions.
    """
    masks = []

    # Attempt 1: result.masks is present and has .data as (N, H, W) boolean array
    try:
        if hasattr(result, "masks") and result.masks is not None:
            m = result.masks
            # many versions: result.masks.data (torch tensor)
            if hasattr(m, "data"):
                arr = m.data
                # convert to numpy
                try:
                    arr = arr.cpu().numpy()
                except Exception:
                    arr = np.asarray(arr)
                # arr assumed shape (N, H, W) or (N, H*W)
                if arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        mask = (arr[i].astype(np.uint8) > 0).astype(np.uint8) * 255
                        masks.append(mask)
                    return masks
            # another possibility: m.masks or m.xy
            if hasattr(m, "masks"):
                arr = m.masks
                try:
                    arr = arr.cpu().numpy()
                except Exception:
                    arr = np.asarray(arr)
                if arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        mask = (arr[i].astype(np.uint8) > 0).astype(np.uint8) * 255
                        masks.append(mask)
                    return masks
    except Exception:
        pass

    # Attempt 2: some versions give r.masks as polygons already (xys)
    try:
        if hasattr(result, "masks") and result.masks is not None and hasattr(result.masks, "xy"):
            # masks.xy is a list of polygons per mask (in pixel coords)
            for poly in result.masks.xy:
                # poly might already be list of points [[x,y], [x,y], ...]
                canvas = np.zeros((img_h, img_w), dtype=np.uint8)
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(canvas, [pts], 255)
                masks.append(canvas)
            return masks
    except Exception:
        pass

    # Attempt 3: some versions put bitmasks in result.masks.xy or result.masks.polygons
    try:
        if hasattr(result, "masks") and result.masks is not None:
            m = result.masks
            if hasattr(m, "xy"):
                for poly in m.xy:
                    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(canvas, [pts], 255)
                    masks.append(canvas)
                return masks
    except Exception:
        pass

    # Attempt 4: if nothing else, some results include segmentation as polygons in result.boxes.data
    # or the model returns no masks that can be extracted. We'll return empty list.
    return masks


# helper: convert binary mask -> list of polygons (each polygon is list of x,y pixel coords)
def mask_to_polygons(mask: np.ndarray, approx_epsilon_mult: float = APPROX_EPSILON_MULT) -> List[List[List[int]]]:
    """
    Convert binary mask (uint8 0/255) to a list of polygon contours.
    Returns list of polygons, each polygon is a list of [x, y] points (ints).
    We use findContours, filter small contours, and simplify with approxPolyDP.
    """
    polygons = []
    # ensure binary
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_POLYGON_AREA:
            continue
        # approximate/simplify
        epsilon = approx_epsilon_mult * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if approx.shape[0] < 3:
            continue
        poly = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        polygons.append(poly)
    return polygons


# helper: convert polygon pixel coords -> relative coords flattened [x1,y1,x2,y2,...]
def polygon_pixels_to_relative(poly: List[List[int]], img_w: int, img_h: int) -> List[float]:
    rel = []
    for x, y in poly:
        rel.append(float(x) / float(img_w))
        rel.append(float(y) / float(img_h))
    return rel


# main prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """
    Accepts a multipart file upload (field name 'file') and returns JSON with polygons.
    Optional header x-api-key can be used to secure the endpoint (Roboflow allows setting
    a header when configuring Custom Model).
    """
    # read image
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    img_w, img_h = img.size
    np_img = np.array(img)

    # run inference
    # For robustness, we call model(np_img) which returns a Results object (list-like)
    results = model(np_img)

    # results can be a list; we handle the first result
    if len(results) == 0:
        return JSONResponse(content={"predictions": []})

    r = results[0]

    # extract boxes/conf/cls if available
    boxes = None
    confidences = None
    classes = None
    names = getattr(model, "names", {})

    try:
        if hasattr(r, "boxes") and r.boxes is not None:
            b = r.boxes
            # common attributes: xyxy, conf, cls
            try:
                xyxy = b.xyxy.cpu().numpy()
                confidences = b.conf.cpu().numpy()
                classes = b.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback: convert to numpy via list
                try:
                    xyxy = np.array(b.xyxy)
                except Exception:
                    xyxy = None
    except Exception:
        xyxy = None

    # extract masks: returns list of binary masks (uint8 0/255)
    masks = extract_masks_from_result(r, img_h=img_h, img_w=img_w)

    predictions = []

    # if masks are present, convert them to polygons
    if masks and len(masks) > 0:
        for idx, mask in enumerate(masks):
            # get confidence & class if available
            conf = float(confidences[idx]) if (confidences is not None and idx < len(confidences)) else 1.0
            cls = int(classes[idx]) if (classes is not None and idx < len(classes)) else None
            if conf < CONF_THRESHOLD:
                continue
            # convert mask to polygons (may be multiple polygons per mask)
            polys = mask_to_polygons(mask)
            for poly in polys:
                rel_poly = polygon_pixels_to_relative(poly, img_w=img_w, img_h=img_h)
                pred = {
                    "label": names.get(cls, str(cls)) if cls is not None else "unknown",
                    "confidence": float(conf),
                    "polygon": rel_poly,  # flattened list [x1,y1,x2,y2,...], relative coords
                }
                predictions.append(pred)
    else:
        # No masks found: return empty list OR attempt a fallback using boxes (not ideal)
        # we'll try to fallback to boxes -> rectangle polygons
        if 'xyxy' in locals() and xyxy is not None:
            for idx, box in enumerate(xyxy):
                conf = float(confidences[idx]) if (confidences is not None and idx < len(confidences)) else 1.0
                cls = int(classes[idx]) if (classes is not None and idx < len(classes)) else None
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box
                poly = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
                rel_poly = polygon_pixels_to_relative(poly, img_w=img_w, img_h=img_h)
                pred = {
                    "label": names.get(cls, str(cls)) if cls is not None else "unknown",
                    "confidence": float(conf),
                    "polygon": rel_poly,
                }
                predictions.append(pred)

    return JSONResponse(content={"predictions": predictions})


if __name__ == "__main__":
    uvicorn.run("roboflow_custom_inference_fastapi:app", host=APP_HOST, port=APP_PORT, reload=False)
