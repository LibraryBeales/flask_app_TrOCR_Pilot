"""
Image preprocessing module for historical document OCR
Provides multiple preprocessing strategies for bleed-through removal,
contrast enhancement, and binarisation
"""
import cv2
import numpy as np
import os
from typing import Dict, Tuple


def get_best_channel(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """Extract the colour channel with highest contrast"""
    b, g, r = cv2.split(img)
    scores = {
        'blue':  float(np.std(b)),
        'green': float(np.std(g)),
        'red':   float(np.std(r))
    }
    best = max(scores, key=scores.get)
    channel_map = {'blue': b, 'green': g, 'red': r}
    return channel_map[best], best


def preprocess_conservative(img_input) -> np.ndarray:
    """
    Conservative enhancement.
    Best for: mostly readable documents needing gentle cleanup.
    Steps: best channel extraction, denoise, CLAHE, gentle sharpen.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    grey, channel = get_best_channel(img)

    denoised = cv2.fastNlMeansDenoising(
        grey, h=10, templateWindowSize=7, searchWindowSize=21
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened


def preprocess_sauvola(img_input) -> np.ndarray:
    """
    Sauvola binarisation.
    Best for: heavy bleed-through, uneven lighting.
    Produces pure black and white output.
    """
    from skimage.filters import threshold_sauvola

    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    grey, _ = get_best_channel(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    background = cv2.morphologyEx(grey, cv2.MORPH_DILATE, kernel)
    normalised = cv2.divide(grey, background, scale=255)

    thresh = threshold_sauvola(normalised, window_size=25, k=0.2)
    binary = (normalised > thresh).astype(np.uint8) * 255

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    return cleaned


def preprocess_adaptive(img_input) -> np.ndarray:
    """
    Adaptive threshold binarisation.
    Best for: moderate bleed-through, good middle ground.
    Faster than Sauvola.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    grey, _ = get_best_channel(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    background = cv2.morphologyEx(grey, cv2.MORPH_DILATE, kernel)
    normalised = cv2.divide(grey, background, scale=255)

    binary = cv2.adaptiveThreshold(
        normalised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    return cleaned


def preprocess_nobleed(img_input) -> np.ndarray:
    """
    Bleed-through specific removal using channel arithmetic.
    Best for: documents where bleed-through is the dominant problem.
    Bleed-through is warmer (more red) than front-side ink.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    result = np.clip(b * 1.4 - r * 0.4, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    result = clahe.apply(result)

    return result


def preprocess_original_grey(img_input) -> np.ndarray:
    """Simple greyscale conversion for comparison baseline"""
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Registry of all available methods
METHODS = {
    'original':    ('Original Greyscale',    preprocess_original_grey),
    'conservative':('Conservative',           preprocess_conservative),
    'sauvola':     ('Sauvola Binarisation',   preprocess_sauvola),
    'adaptive':    ('Adaptive Threshold',     preprocess_adaptive),
    'nobleed':     ('Bleed-Through Removal',  preprocess_nobleed),
}


def run_all_methods(image_path: str, output_dir: str) -> Dict[str, str]:
    """
    Run all preprocessing methods on one image.
    Returns dict of method_name -> output_filepath.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    results = {}

    for method_key, (method_name, func) in METHODS.items():
        try:
            result = func(image_path)
            out_filename = f"{base}_{method_key}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, result)
            results[method_key] = {
                'filename': out_filename,
                'label':    method_name,
                'success':  True
            }
        except Exception as e:
            results[method_key] = {
                'filename': None,
                'label':    method_name,
                'success':  False,
                'error':    str(e)
            }

    return results


def preprocess_single(image_path: str, method: str, output_dir: str) -> str:
    """
    Run one preprocessing method on one image.
    Returns output filepath.
    """
    os.makedirs(output_dir, exist_ok=True)

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(METHODS.keys())}")

    method_name, func = METHODS[method]
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_filename = f"{base}_{method}.jpg"
    out_path = os.path.join(output_dir, out_filename)

    result = func(image_path)
    cv2.imwrite(out_path, result)
    return out_path