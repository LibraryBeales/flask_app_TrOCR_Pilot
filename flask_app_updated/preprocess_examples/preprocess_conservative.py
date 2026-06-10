# Save as preprocess_conservative.py
import cv2
import numpy as np
from PIL import Image
import os

def preprocess_conservative(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Conservative preprocessing pipeline:
    1. Extract best colour channel
    2. Denoise
    3. CLAHE contrast enhancement
    4. Gentle sharpening
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + '_clean' + ext

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 1 — Extract the best channel for ink visibility
    # For aged documents on yellowed paper, the blue channel
    # often has the highest contrast between ink and paper
    b, g, r = cv2.split(img)

    # Score each channel by its standard deviation (higher = more contrast)
    scores = {
        'blue':  np.std(b),
        'green': np.std(g),
        'red':   np.std(r)
    }
    best_channel = max(scores, key=scores.get)
    channel_map  = {'blue': b, 'green': g, 'red': r}
    grey = channel_map[best_channel]
    print(f"Best channel: {best_channel} (scores: {scores})")

    # Step 2 — Denoise while preserving edges
    # h=10 is gentle, increase to 15-20 if bleed-through is heavy
    denoised = cv2.fastNlMeansDenoising(grey, h=10, templateWindowSize=7,
                                         searchWindowSize=21)

    # Step 3 — CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    # Improves local contrast without blowing out highlights
    # clipLimit controls how aggressively contrast is enhanced
    # tileGridSize controls the local region size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 4 — Gentle unsharp mask to sharpen letter edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Save result
    cv2.imwrite(output_path, sharpened)
    print(f"Saved to: {output_path}")
    return sharpened


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_conservative.py image.jpg")
        sys.exit(1)
    preprocess_conservative(sys.argv[1])