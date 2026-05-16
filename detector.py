import contextlib
import os
import sys

if sys.platform == "darwin":
    _homebrew_lib = "/opt/homebrew/lib"
    if os.path.isdir(_homebrew_lib):
        os.environ.setdefault("DYLD_LIBRARY_PATH", _homebrew_lib)

import cv2
import numpy as np
from pyzbar import pyzbar

from preprocessor import preprocess


@contextlib.contextmanager
def _suppress_c_stderr():
    """Suppress C-level stderr (zbar assertion warnings)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)

_cv2_qr = cv2.QRCodeDetector()

_ROTATION_CODES = [
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
]

ROTATION_CATEGORIES = {"rotations", "perspective"}
CV2_FIRST_CATEGORIES = {"high_version", "curved", "damaged"}


WARP_CATEGORIES = {"high_version", "curved", "damaged", "perspective", "blurred",
                   "bright_spots", "nominal", "rotations", "glare", "close",
                   "monitor", "noncompliant", "shadows", "pathological", "brightness"}


def _decode_objects(decoded_objects) -> list[dict]:
    results = []
    for obj in decoded_objects:
        results.append({
            "data": obj.data.decode("utf-8", errors="replace"),
            "type": obj.type,
            "rect": {
                "x": obj.rect.left,
                "y": obj.rect.top,
                "w": obj.rect.width,
                "h": obj.rect.height,
            },
            "polygon": [(p.x, p.y) for p in obj.polygon],
        })
    return results


def _decode_cv2qr(gray: np.ndarray) -> list[dict]:
    data, points, _ = _cv2_qr.detectAndDecode(gray)
    if not data or points is None or len(points) == 0:
        return []
    pts = points[0] if points.ndim == 3 else points
    xs = pts[:, 0]
    ys = pts[:, 1]
    return [{
        "data": data,
        "type": "QRCODE",
        "rect": {
            "x": int(xs.min()),
            "y": int(ys.min()),
            "w": int(xs.max() - xs.min()),
            "h": int(ys.max() - ys.min()),
        },
        "polygon": [(int(p[0]), int(p[1])) for p in pts],
    }]


def _decode_cv2qr_multi(gray: np.ndarray) -> list[dict]:
    ok, decoded_info, points_multi, _ = _cv2_qr.detectAndDecodeMulti(gray)
    if not ok or not decoded_info:
        return []
    for i, info in enumerate(decoded_info):
        if info and points_multi is not None and i < len(points_multi):
            pts = points_multi[i]
            xs = pts[:, 0]
            ys = pts[:, 1]
            return [{
                "data": info,
                "type": "QRCODE",
                "rect": {
                    "x": int(xs.min()),
                    "y": int(ys.min()),
                    "w": int(xs.max() - xs.min()),
                    "h": int(ys.max() - ys.min()),
                },
                "polygon": [(int(p[0]), int(p[1])) for p in pts],
            }]
    return []


def _try_warp_decode(gray: np.ndarray) -> list[dict]:
    ok, points = _cv2_qr.detect(gray)
    if not ok or points is None:
        return []

    pts = points[0] if points.ndim == 3 else points
    if len(pts) < 4:
        return []

    # Try detect+decode separately on raw and CLAHE
    for img in [gray, cv2.createCLAHE(clipLimit=3.0).apply(gray)]:
        try:
            data, _ = _cv2_qr.decode(img, points)
            if data:
                xs = pts[:, 0]
                ys = pts[:, 1]
                return [{
                    "data": data,
                    "type": "QRCODE",
                    "rect": {
                        "x": int(xs.min()), "y": int(ys.min()),
                        "w": int(xs.max() - xs.min()), "h": int(ys.max() - ys.min()),
                    },
                    "polygon": [(int(p[0]), int(p[1])) for p in pts],
                }]
        except Exception:
            pass

    # Crop to QR region with padding, try multiple scales
    xs = pts[:, 0]
    ys = pts[:, 1]
    pad = 30
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(gray.shape[1], int(xs.max()) + pad)
    y2 = min(gray.shape[0], int(ys.max()) + pad)
    cropped = gray[y1:y2, x1:x2]
    ch, cw = cropped.shape[:2]

    if ch >= 50 and cw >= 50:
        for target in [500, 800, 1000, 1500, 2000]:
            s = target / max(ch, cw)
            if s < 0.5:
                continue
            resized = cv2.resize(cropped, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for img in [resized, cv2.createCLAHE(clipLimit=3.0).apply(resized)]:
                with _suppress_c_stderr():
                    results = _decode_objects(pyzbar.decode(img))
                if results:
                    return results
                results = _decode_cv2qr(img)
                if results:
                    return results

    # Perspective warp
    pts_f = pts[:4].astype(np.float32)
    s = pts_f.sum(axis=1)
    d_pts = np.diff(pts_f, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts_f[np.argmin(s)]
    ordered[2] = pts_f[np.argmax(s)]
    ordered[1] = pts_f[np.argmin(d_pts)]
    ordered[3] = pts_f[np.argmax(d_pts)]

    w1 = np.linalg.norm(ordered[1] - ordered[0])
    w2 = np.linalg.norm(ordered[2] - ordered[3])
    h1 = np.linalg.norm(ordered[3] - ordered[0])
    h2 = np.linalg.norm(ordered[2] - ordered[1])
    out_w = int(max(w1, w2))
    out_h = int(max(h1, h2))
    if out_w < 50 or out_h < 50:
        return []

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h))

    for img in [warped, cv2.createCLAHE(clipLimit=3.0).apply(warped)]:
        with _suppress_c_stderr():
            results = _decode_objects(pyzbar.decode(img))
        if results:
            return results
        results = _decode_cv2qr(img)
        if results:
            return results

    # Try upscaled warped image
    for scale in [2.0, 3.0]:
        big = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        with _suppress_c_stderr():
            results = _decode_objects(pyzbar.decode(big))
        if results:
            return results
        results = _decode_cv2qr(big)
        if results:
            return results

    return []


def decode_qr(image_path: str) -> list[dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    with _suppress_c_stderr():
        return _decode_objects(pyzbar.decode(gray))


def decode_qr_adaptive(image_path: str, category: str | None = None) -> list[dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if category is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with _suppress_c_stderr():
            return _decode_objects(pyzbar.decode(gray))

    candidates = preprocess(image, category)
    do_rotate = category in ROTATION_CATEGORIES
    cv2_first = category in CV2_FIRST_CATEGORIES

    def _run_pass(decode_fn, label):
        for candidate in candidates:
            if do_rotate:
                for code in _ROTATION_CODES:
                    img = cv2.rotate(candidate, code) if code is not None else candidate
                    results = decode_fn(img)
                    if results:
                        return results
            else:
                results = decode_fn(candidate)
                if results:
                    return results
        return []

    if cv2_first:
        decode_fn_cv2 = lambda img: _decode_cv2qr(img)
        def decode_fn_pyzbar(img):
            with _suppress_c_stderr():
                return _decode_objects(pyzbar.decode(img))
        results = _run_pass(decode_fn_cv2, "cv2")
        if results:
            return results
        results = _run_pass(decode_fn_pyzbar, "pyzbar")
        if results:
            return results
    else:
        def decode_fn_pyzbar(img):
            with _suppress_c_stderr():
                return _decode_objects(pyzbar.decode(img))
        decode_fn_cv2 = lambda img: _decode_cv2qr(img)
        results = _run_pass(decode_fn_pyzbar, "pyzbar")
        if results:
            return results
        results = _run_pass(decode_fn_cv2, "cv2")
        if results:
            return results

    if category in WARP_CATEGORIES:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = _try_warp_decode(gray)
        if results:
            return results

    # Last resort: detectAndDecodeMulti on key preprocessed variants
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe3 = cv2.createCLAHE(clipLimit=3.0).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    multi_candidates = [gray, clahe3, otsu]
    h, w = gray.shape[:2]
    for d in [800, 1200]:
        cur = max(h, w)
        if cur > d:
            s = d / cur
            multi_candidates.append(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    for img in multi_candidates:
        results = _decode_cv2qr_multi(img)
        if results:
            return results

    return []


def draw_results(image: np.ndarray, results: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for r in results:
        pts = r["polygon"]
        for i in range(len(pts)):
            cv2.line(canvas, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 3)

        x, y = r["rect"]["x"], r["rect"]["y"]
        cv2.putText(
            canvas,
            r["data"],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return canvas
