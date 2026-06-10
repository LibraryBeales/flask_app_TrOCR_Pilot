# Save as preprocess_binarise.py
import cv2
import numpy as np
import os

def preprocess_binarise(image_path: str, output_path: str = None,
                         method: str = 'sauvola') -> np.ndarray:
    """
    Aggressive binarisation pipeline:
    1. Channel extraction
    2. Background estimation and removal
    3. Sauvola or Otsu binarisation
    4. Morphological cleanup

    method: 'sauvola' (best for uneven lighting) or 'otsu' (faster, simpler)
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + '_binary' + ext

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 1 — Convert to greyscale using weighted channel blend
    # This weights channels to suppress bleed-through
    # Bleed-through ink is usually reddish-brown so reducing red helps
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Alternative: use only blue channel which suppresses warm-toned bleed
    # grey = cv2.split(img)[0]

    # Step 2 — Background normalisation using morphological opening
    # This estimates the background illumination and removes it
    # kernel size should be larger than the tallest letter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    background = cv2.morphologyEx(grey, cv2.MORPH_DILATE, kernel)
    normalised = cv2.divide(grey, background, scale=255)

    # Step 3 — Binarisation
    if method == 'sauvola':
        # Sauvola is best for historical documents with uneven lighting
        # Requires scikit-image
        from skimage.filters import threshold_sauvola
        # window_size should be roughly the size of a character
        # k controls sensitivity — lower = more aggressive
        thresh = threshold_sauvola(normalised, window_size=25, k=0.2)
        binary = (normalised > thresh).astype(np.uint8) * 255

    elif method == 'otsu':
        # Otsu is simpler and faster but assumes even lighting
        _, binary = cv2.threshold(
            normalised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    elif method == 'adaptive':
        # Adaptive threshold — good middle ground
        # blockSize must be odd, C is subtracted from mean
        binary = cv2.adaptiveThreshold(
            normalised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    # Step 4 — Morphological cleanup
    # Remove small noise specks (bleed-through dots)
    # kernel size controls minimum feature size to keep
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    # Optional: close small gaps in letters
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    cv2.imwrite(output_path, cleaned)
    print(f"Saved to: {output_path}")
    return cleaned


if __name__ == '__main__':
    import sys
    method = sys.argv[2] if len(sys.argv) > 2 else 'sauvola'
    if len(sys.argv) < 2:
        print("Usage: python preprocess_binarise.py image.jpg [sauvola|otsu|adaptive]")
        sys.exit(1)
    preprocess_binarise(sys.argv[1], method=method)