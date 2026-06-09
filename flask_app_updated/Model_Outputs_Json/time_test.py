# Save as test_timing.py
import time
import os
import sys

PROJECT_ROOT = r"C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR"
APP_DIR = os.path.join(PROJECT_ROOT, "flask_app_updated")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from modular_app.config import get_config
from modular_app.ocr_models import OCRModelsService
from modular_app.llm_service import LLMService

config = get_config()
base_dir = APP_DIR
config.upload_folder = os.path.join(base_dir, config.upload_folder)
config.line_segments_folder = os.path.join(base_dir, config.line_segments_folder)
config.inference_folder = os.path.join(base_dir, config.inference_folder)
config.annotation_folder = os.path.join(base_dir, config.annotation_folder)

# Use one of your existing uploaded images
IMAGE_PATH = os.path.join(
    config.upload_folder,
    os.listdir(config.upload_folder)[0]
)
print(f"Timing test on: {os.path.basename(IMAGE_PATH)}")

ocr_service = OCRModelsService(config)
ocr_service.load_trocr_model()
llm_service = LLMService(config)

# Time the full pipeline
total_start = time.time()

# Step 1 - Kraken segmentation
t1 = time.time()
boxes, scores, image = ocr_service.extract_textlines_from_image(IMAGE_PATH)
t2 = time.time()
print(f"Kraken segmentation: {t2-t1:.1f}s — {len(boxes)} lines found")

# Step 2 - TrOCR recognition
t3 = time.time()
from modular_app.ocr_models import OCRModelsService
import cv2
import numpy as np
crops, padded = ocr_service.crop_textlines_with_padding(image, boxes)
results = ocr_service.process_textlines_with_trocr(crops)
t4 = time.time()
print(f"TrOCR recognition: {t4-t3:.1f}s — {len(results)} lines read")

# Step 3 - LLM correction
t5 = time.time()
segments = [
    {'line_index': i, 'ocr_text': r['text'], 'ocr_text_pre_llm': r['text']}
    for i, r in enumerate(results)
]
corrected = llm_service.process_line_segments_with_gemini(segments)
t6 = time.time()
print(f"LLM correction: {t6-t5:.1f}s — {len(corrected)} lines corrected")

total = time.time() - total_start
print(f"")
print(f"Total per page: {total:.1f}s")
print(f"Estimated 1000 pages: {total * 1000 / 3600:.1f} hours")
print(f"Estimated 1000 pages (parallel x2 GPU): {total * 1000 / 3600 / 2:.1f} hours")