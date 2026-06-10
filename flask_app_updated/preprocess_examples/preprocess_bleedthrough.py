# Save as preprocess_bleedthrough.py
import cv2
import numpy as np
import os

def remove_bleedthrough(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Bleed-through specific removal using colour channel arithmetic.

    Bleed-through text is typically:
    - Lighter than the front-side ink
    - Warmer in colour (more red/yellow)
    - Lower contrast than front-side text

    Strategy: use the difference between colour channels to
    separate front-side ink from bleed-through.
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + '_nobleed' + ext

    img = cv2.imread(image_path).astype(np.float32)
    b, g, r = cv2.split(img)

    # Front-side dark ink appears in all channels equally
    # Bleed-through appears more in red/green than blue
    # Subtracting red from blue isolates front-side ink

    # Method A — Channel difference
    # Amplify blue, suppress red to kill warm bleed-through
    result = np.clip(b * 1.4 - r * 0.4, 0, 255).astype(np.uint8)

    # Method B — Use minimum of channels
    # Front-side ink is dark in ALL channels
    # Bleed-through is only dark in some channels
    # min_channel = np.min(cv2.split(img), axis=0).astype(np.uint8)

    # Enhance contrast after bleed removal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    result = clahe.apply(result)

    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_bleedthrough.py image.jpg")
        sys.exit(1)
    remove_bleedthrough(sys.argv[1])