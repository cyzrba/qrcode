import cv2
import numpy as np


def apply_clahe(gray: np.ndarray, clip_limit=2.0, grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


def apply_unsharp_mask(gray: np.ndarray, sigma=1.5, strength=1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def apply_adaptive_threshold(gray: np.ndarray, block_size=11, C=2) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)


def apply_morphological_close(gray: np.ndarray, kernel_size=3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def apply_morphological_open(gray: np.ndarray, kernel_size=3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


def apply_sauvola(gray: np.ndarray, window_size=25, k=0.2) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (window_size, window_size))
    sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    return ((gray_f > threshold) * 255).astype(np.uint8)


def apply_bilateral_filter(gray: np.ndarray, d=9, sigma_color=75, sigma_space=75) -> np.ndarray:
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def apply_otsu(gray: np.ndarray) -> np.ndarray:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def apply_dilate(gray: np.ndarray, kernel_size=2, iterations=1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(gray, kernel, iterations=iterations)


def apply_erode(gray: np.ndarray, kernel_size=2, iterations=1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(gray, kernel, iterations=iterations)


def _resize_gray(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_dim:
        return gray
    scale = max_dim / cur_max
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _multi_scale(gray: np.ndarray, dims: list[int]) -> list[np.ndarray]:
    candidates = []
    for d in dims:
        r = _resize_gray(gray, d)
        if not any(r is c or (r.shape == c.shape and np.array_equal(r, c)) for c in candidates):
            candidates.append(r)
    return candidates


def _preprocess_nominal(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [apply_clahe(gray), gray, apply_clahe(gray, clip_limit=3.0)]


def _preprocess_blurred(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharp2 = apply_unsharp_mask(gray, sigma=2.0, strength=2.0)
    sharp1 = apply_unsharp_mask(gray, sigma=1.0, strength=1.5)
    sharp3 = apply_unsharp_mask(gray, sigma=3.0, strength=2.5)
    bilateral = apply_bilateral_filter(gray)
    return [
        apply_clahe(sharp2),
        apply_clahe(sharp1),
        apply_clahe(sharp3),
        apply_clahe(bilateral),
        apply_clahe(apply_unsharp_mask(bilateral, sigma=2.0, strength=2.0)),
        apply_sauvola(sharp2, 25, 0.2),
        apply_otsu(gray),
        apply_dilate(gray),
        apply_erode(gray),
    ]


def _preprocess_bright_spots(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    corrected = cv2.subtract(gray, blackhat)
    return [
        apply_clahe(corrected),
        apply_clahe(gray),
        apply_clahe(corrected, clip_limit=3.0),
        apply_sauvola(apply_clahe(corrected), 25, 0.2),
        apply_otsu(gray),
        apply_dilate(gray),
        apply_erode(gray),
    ]


def _preprocess_brightness(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = apply_gamma(gray, 0.5)
    bright = apply_gamma(gray, 2.0)
    return [apply_clahe(dark), apply_clahe(bright), apply_clahe(gray)]


def _preprocess_close(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for d in [300, 500, 700, 1000, 1500, 2000]:
        r = _resize_gray(gray, d)
        candidates.append(r)
        candidates.append(apply_clahe(r))
    return candidates


def _preprocess_curved(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray, clip_limit=3.0)
    clahe4 = apply_clahe(gray, clip_limit=4.0)
    sharp = apply_unsharp_mask(gray, sigma=1.5, strength=2.0)
    return [
        clahe,
        apply_adaptive_threshold(clahe, 21, 3),
        apply_adaptive_threshold(clahe, 31, 5),
        apply_adaptive_threshold(clahe, 41, 7),
        apply_clahe(sharp),
        apply_adaptive_threshold(clahe4, 21, 3),
        apply_sauvola(clahe, 25, 0.2),
        apply_sauvola(clahe, 31, 0.15),
        apply_otsu(clahe),
        apply_otsu(gray),
        apply_dilate(gray),
        apply_erode(gray),
    ]


def _preprocess_damaged(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray)
    candidates = []
    for d in [800, 1000, 500, 1500, 2000, 300]:
        r = _resize_gray(clahe, d)
        candidates.append(r)
    for d in [800, 1000, 500, 1500]:
        r = _resize_gray(clahe, d)
        candidates.append(apply_adaptive_threshold(r, 21, 3))
    for d in [800, 1000]:
        r = _resize_gray(clahe, d)
        candidates.append(apply_adaptive_threshold(r, 31, 5))
    closed = apply_morphological_close(clahe, 3)
    candidates.append(closed)
    candidates.append(apply_adaptive_threshold(closed, 21, 3))
    opened = apply_morphological_open(clahe, 3)
    candidates.append(apply_adaptive_threshold(opened, 21, 3))
    candidates.append(apply_sauvola(clahe, 25, 0.2))
    for d in [800, 1000]:
        r = _resize_gray(clahe, d)
        candidates.append(apply_sauvola(r, 25, 0.2))
    sharp = apply_unsharp_mask(gray, sigma=1.5, strength=2.0)
    candidates.append(apply_clahe(sharp))
    candidates.append(apply_otsu(clahe))
    candidates.append(apply_otsu(gray))
    candidates.append(apply_dilate(gray))
    candidates.append(apply_erode(gray))
    return candidates


def _preprocess_glare(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    inpainted = cv2.inpaint(gray, mask, 10, cv2.INPAINT_TELEA)
    cl_inpaint = apply_clahe(inpainted)
    cl_raw = apply_clahe(gray)
    candidates = []
    for d in [800, 1000, 500, 1500, 2000]:
        r = _resize_gray(gray, d)
        candidates.append(r)
    for d in [800, 1000, 500, 1500]:
        candidates.append(_resize_gray(cl_raw, d))
        candidates.append(_resize_gray(cl_inpaint, d))
    candidates.append(apply_sauvola(cl_inpaint, 25, 0.2))
    return candidates


def _preprocess_high_version(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for d in [3000, 2500, 2000, 1500, 1200, 800, 600, 400]:
        r = _resize_gray(gray, d)
        candidates.append(r)
    candidates.append(apply_clahe(gray))
    for d in [2000, 1500, 1200]:
        r = _resize_gray(gray, d)
        candidates.append(apply_clahe(r))
    candidates.append(apply_sauvola(apply_clahe(gray), 25, 0.2))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    for d in [2000, 1500]:
        r = _resize_gray(blurred, d)
        candidates.append(r)
    candidates.append(apply_clahe(blurred, clip_limit=3.0))
    for d in [2000, 1500]:
        r = _resize_gray(gray, d)
        candidates.append(apply_adaptive_threshold(r, 21, 3))
    candidates.append(apply_otsu(gray))
    candidates.append(apply_dilate(gray))
    candidates.append(apply_erode(gray))
    return candidates


def _preprocess_lots(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [apply_clahe(gray), gray]


def _preprocess_monitor(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for d in [300, 500, 800, 1000, 1500, 2000]:
        r = _resize_gray(gray, d)
        candidates.append(r)
        candidates.append(apply_clahe(r, clip_limit=3.0))
    return candidates


def _preprocess_noncompliant(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray)
    return [
        clahe,
        apply_adaptive_threshold(clahe, 11, 2),
        apply_adaptive_threshold(clahe, 21, 3),
        apply_adaptive_threshold(clahe, 31, 5),
        apply_sauvola(clahe, 25, 0.2),
        apply_clahe(gray, clip_limit=3.0),
    ]


def _preprocess_pathological(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray, clip_limit=3.0)
    sharp = apply_unsharp_mask(gray, sigma=1.5, strength=2.0)
    bilateral = apply_bilateral_filter(gray)
    return [
        gray,
        clahe,
        apply_clahe(sharp),
        apply_adaptive_threshold(clahe, 11, 2),
        apply_adaptive_threshold(clahe, 21, 3),
        apply_clahe(bilateral),
        apply_sauvola(clahe, 25, 0.2),
    ]


def _preprocess_perspective(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray)
    clahe3 = apply_clahe(gray, clip_limit=3.0)
    sharp = apply_unsharp_mask(gray, sigma=1.5, strength=2.0)
    return [
        clahe,
        apply_adaptive_threshold(clahe, 21, 3),
        apply_adaptive_threshold(clahe, 31, 5),
        apply_adaptive_threshold(clahe3, 21, 3),
        apply_adaptive_threshold(clahe3, 31, 5),
        apply_clahe(sharp),
        apply_sauvola(clahe, 25, 0.2),
        apply_otsu(clahe),
        apply_dilate(gray),
        apply_erode(gray),
    ]


def _preprocess_rotations(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [
        apply_clahe(gray),
        apply_clahe(gray, clip_limit=3.0),
        gray,
        apply_unsharp_mask(gray, sigma=1.5, strength=1.5),
    ]


def _preprocess_shadows(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe3 = apply_clahe(gray, clip_limit=3.0)
    return [
        clahe3,
        apply_clahe(gray),
        apply_sauvola(clahe3, 25, 0.2),
    ]


_PREPROCESSORS = {
    "nominal": _preprocess_nominal,
    "blurred": _preprocess_blurred,
    "bright_spots": _preprocess_bright_spots,
    "brightness": _preprocess_brightness,
    "close": _preprocess_close,
    "curved": _preprocess_curved,
    "damaged": _preprocess_damaged,
    "glare": _preprocess_glare,
    "high_version": _preprocess_high_version,
    "lots": _preprocess_lots,
    "monitor": _preprocess_monitor,
    "noncompliant": _preprocess_noncompliant,
    "pathological": _preprocess_pathological,
    "perspective": _preprocess_perspective,
    "rotations": _preprocess_rotations,
    "shadows": _preprocess_shadows,
}


def preprocess(image: np.ndarray, category: str) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fn = _PREPROCESSORS.get(category)
    if fn is None:
        return [gray]
    candidates = fn(image)
    has_gray = any(
        c.shape == gray.shape and np.array_equal(c, gray)
        for c in candidates
    )
    if not has_gray:
        candidates.append(gray)
    return candidates
