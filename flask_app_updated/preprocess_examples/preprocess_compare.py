# Save as preprocess_compare.py
import cv2
import numpy as np
import os
import sys

# Import the three workflows
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_conservative import preprocess_conservative
from preprocess_binarise import preprocess_binarise
from preprocess_bleedthrough import remove_bleedthrough

def compare_all(image_path: str):
    """Run all preprocessing methods and save comparison image"""

    base, ext = os.path.splitext(image_path)

    print("Running all preprocessing methods...")

    # Run each method
    original  = cv2.imread(image_path)
    grey      = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    conserv   = preprocess_conservative(image_path, base + '_1_conservative' + ext)
    sauvola   = preprocess_binarise(image_path, base + '_2_sauvola' + ext, 'sauvola')
    adaptive  = preprocess_binarise(image_path, base + '_3_adaptive' + ext, 'adaptive')
    nobleed   = remove_bleedthrough(image_path, base + '_4_nobleed' + ext)

    # Resize all to same height for comparison
    target_h = 400
    def resize_h(img, h):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    images = [
        ('Original Grey', resize_h(grey, target_h)),
        ('1 Conservative', resize_h(conserv, target_h)),
        ('2 Sauvola', resize_h(sauvola, target_h)),
        ('3 Adaptive', resize_h(adaptive, target_h)),
        ('4 No Bleed', resize_h(nobleed, target_h)),
    ]

    # Add labels
    labelled = []
    for label, img in images:
        labelled_img = np.zeros((target_h + 30, img.shape[1]), dtype=np.uint8)
        labelled_img[30:, :] = img
        cv2.putText(labelled_img, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)
        labelled.append(labelled_img)

    # Stack horizontally
    comparison = np.hstack(labelled)
    output_path = base + '_COMPARISON' + ext
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved to: {output_path}")
    print("Open the COMPARISON file to decide which method works best.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python preprocess_compare.py image.jpg")
        sys.exit(1)
    compare_all(sys.argv[1])