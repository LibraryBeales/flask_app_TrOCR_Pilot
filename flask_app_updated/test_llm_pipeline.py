# Save as test_llm_pipeline.py
import os
import sys

PROJECT_ROOT = r"C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR"
APP_DIR = os.path.join(PROJECT_ROOT, "flask_app_updated")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from modular_app.config import get_config
from modular_app.llm_service import LLMService

config = get_config()
llm = LLMService(config)

# Test 1 - Is Ollama reachable
print("=" * 50)
print("TEST 1 - Ollama reachable")
result = llm.init_local_llama()
print(f"Ollama available: {result}")

# Test 2 - Try a single line correction
print("=" * 50)
print("TEST 2 - Single line correction")
test_text = "MME Burton S. Lifson"
corrected, status = llm.try_local_llama_correction(test_text)
print(f"Input:   {test_text}")
print(f"Output:  {corrected}")
print(f"Status:  {status}")

# Test 3 - Try the full fallback chain
print("=" * 50)
print("TEST 3 - Full fallback chain")
result, status = llm.process_text_with_fallbacks(test_text)
print(f"Output:  {result}")
print(f"Status:  {status}")

# Test 4 - Try process_line_segments_with_gemini
print("=" * 50)
print("TEST 4 - process_line_segments_with_gemini")
test_segments = [
    {'line_index': 0, 'ocr_text': 'MME Burton S. Lifson', 'ocr_text_pre_llm': 'MME Burton S. Lifson'},
    {'line_index': 1, 'ocr_text': 'BRANCH OF SERVICE Army Force', 'ocr_text_pre_llm': 'BRANCH OF SERVICE Army Force'}
]
corrected_segments = llm.process_line_segments_with_gemini(test_segments)
for seg in corrected_segments:
    print(f"Line {seg['line_index']}:")
    print(f"  ocr_text:           {seg.get('ocr_text', 'MISSING')}")
    print(f"  ocr_text_corrected: {seg.get('ocr_text_corrected', 'MISSING')}")
    print(f"  correction_status:  {seg.get('correction_status', 'MISSING')}")

print("=" * 50)