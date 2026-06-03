"""
LLM Service - Handles text correction using multiple backends with fallback system
Supports Gemini, HuggingFace, and Local LLaMA with automatic fallback
"""
import os
import time
import logging
import requests
from typing import Tuple, List, Dict, Optional

# Optional imports with guards
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("Gemini API integration available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini API not available - install google-generativeai")

LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMService:
    """Service for text correction using multiple LLM backends"""
    
    def __init__(self, config):
        self.config = config
        self.local_llama_client = None
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Setup Gemini API if available"""
        if GEMINI_AVAILABLE and self.config.gemini_api_key:
            try:
                genai.configure(api_key=self.config.gemini_api_key)
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.error(f"Error configuring Gemini API: {e}")
        else:
            logger.info("Gemini API not available - missing key or library")
    
    def chunk_text(self, text: str, chunk_size_chars: int = None) -> List[str]:
        """Split text into chunks if it exceeds threshold"""
        if chunk_size_chars is None:
            chunk_size_chars = self.config.fallback_chunk_size
        
        if len(text) <= chunk_size_chars:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size_chars):
            chunks.append(text[i:i + chunk_size_chars])
        
        return chunks
    
    def create_correction_prompt(self, text: str, context: str = "") -> str:
        """Create structured prompt for text correction with context"""
        context_section = ""
        if context:
            context_section = f"""
Previous context:
{context}

"""
        
        return f"""
Correct the following Spanish OCR text while preserving original grammar and style.
Only fix orthographic errors, punctuation, and obvious OCR mistakes.
{context_section}
Text to correct:
{text}

Instructions:
- Fix spelling errors and OCR artifacts
- Preserve historical language patterns  
- Maintain original formatting
- Return ONLY the corrected text

Corrected text:
"""
    
    def try_gemini_correction(self, text: str, context: str = "", retries: int = 2) -> Tuple[str, str]:
        """Try Gemini API for text correction"""
        if not GEMINI_AVAILABLE or not self.config.gemini_api_key:
            return text, "gemini_unavailable"
        
        prompt = self.create_correction_prompt(text, context)
        
        for attempt in range(retries):
            try:
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
                if response.candidates and response.text:
                    return response.text.strip(), "gemini"
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return text, "gemini_failed"
    
    def try_huggingface_correction(self, text: str, context: str = "", retries: int = 2) -> Tuple[str, str]:
        """Try Hugging Face Inference API for text correction"""
        if not self.config.hf_api_token:
            return text, "hf_unavailable"
        
        prompt = self.create_correction_prompt(text, context)
        
        # Use a general text generation model
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {self.config.hf_api_token}"}
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers,
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
                    timeout=self.config.llm_timeout_seconds
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                        if generated_text and generated_text != prompt:
                            return generated_text.replace(prompt, '').strip(), "hf"
                
            except Exception as e:
                logger.warning(f"HuggingFace attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return text, "hf_failed"
    
    def init_local_llama(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(
                "http://localhost:11434",
                timeout=5
            )
            if "Ollama is running" in response.text:
                logger.info("Ollama service is available")
                return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        return False

    
    def try_local_llama_correction(self, text: str, context: str = "") -> Tuple[str, str]:
        """Try local LLaMA via Ollama for text correction"""
        if not self.init_local_llama():
            return text, "local_llama_unavailable"

        prompt = self.create_correction_prompt(text, context)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": getattr(self.config, 'ollama_model', 'qwen3:32b'),
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.config.llm_timeout_seconds
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                if generated_text:
                    return generated_text, "local_llama"

        except Exception as e:
            logger.error(f"Ollama correction failed: {e}")

        return text, "local_llama_failed"
    
    def process_text_with_fallbacks(self, text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> Tuple[str, str]:
        """
        Main fallback wrapper function that tries all LLM backends in order:
        1. Gemini (primary)
        2. HuggingFace (secondary)  
        3. Local LLaMA (final fallback)
        """
        
        # Try Gemini first
        logger.info("Attempting Gemini correction...")
        result, status = self.try_gemini_correction(text, context, retries)
        if status == "gemini":
            logger.info("Used: gemini")
            return result, status
        
        # Fallback to HuggingFace
        logger.info("Fallback: attempting HuggingFace correction...")
        result, status = self.try_huggingface_correction(text, context, retries)
        if status == "hf":
            logger.info("Fallback: hf") 
            return result, status
        
        # Final fallback to local LLaMA
        logger.info("Fallback: attempting local LLaMA correction...")
        result, status = self.try_local_llama_correction(text, context)
        if status == "local_llama":
            logger.info("Fallback: local_llama")
            return result, status
        
        # All backends failed
        logger.warning("All LLM backends failed, returning original text")
        return text, "all_failed"
    
    def process_text_with_chunking(self, text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> Tuple[str, str]:
        """Process text with chunking support for large inputs"""
        if len(text) <= self.config.fallback_chunk_size:
            return self.process_text_with_fallbacks(text, context, max_tokens, retries)
        
        # Chunk the text
        chunks = self.chunk_text(text, self.config.fallback_chunk_size)
        logger.info(f"Processing {len(chunks)} chunks...")
        
        corrected_chunks = []
        current_context = context
        final_status = "all_failed"
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            corrected_chunk, status = self.process_text_with_fallbacks(chunk, current_context, max_tokens, retries)
            corrected_chunks.append(corrected_chunk)
            
            # Update status to first successful one
            if final_status == "all_failed" and status != "all_failed":
                final_status = status
            
            # Update context for next chunk (last 2 lines)
            chunk_lines = corrected_chunk.strip().split('\n')
            if len(chunk_lines) >= 2:
                current_context = '\n'.join(chunk_lines[-2:])
            elif len(chunk_lines) == 1:
                current_context = chunk_lines[0]
        
        return '\n'.join(corrected_chunks), final_status
    
    def get_last_two_lines(self, text: str) -> str:
        """Extract last two lines from text for context"""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return '\n'.join(lines[-2:])
        elif len(lines) == 1:
            return lines[0]
        return ""
    
    def process_text_with_gemini(self, text: str, context: str = "", max_retries: int = 1) -> Tuple[str, str]:
        """Updated to use new three-tier fallback system"""
        result, status = self.process_text_with_fallbacks(text, context, retries=max_retries)
        
        # Map new status codes to legacy expected return values
        if status in ["gemini", "hf", "local_llama"]:
            return result, "success"
        else:
            return text, "max_retries_exceeded"
    
    def process_line_segments_with_gemini(self, line_segments: List[Dict], context: str = "") -> List[Dict]:
        """Process all line segments in a single batch LLM call"""
        import re

        # Separate empty and non-empty lines
        non_empty = [
            s for s in line_segments
            if s.get('ocr_text', '').strip()
        ]

        if not non_empty:
            for segment in line_segments:
                segment['ocr_text_corrected'] = segment.get('ocr_text', '')
                segment['correction_status'] = 'empty'
            return line_segments

        # Build single prompt with all lines numbered
        lines_text = '\n'.join([
            f"{i+1}. {s.get('ocr_text', '')}"
            for i, s in enumerate(non_empty)
        ])

        prompt = f"""You are correcting OCR text from a scanned historical document.
    Below are {len(non_empty)} lines of OCR output, each numbered.
    Fix only clear OCR errors, spelling mistakes, and garbled characters.
    Preserve the original language, style, and formatting.
    Return ONLY the corrected lines in the same numbered format.
    Do not add explanations or commentary.

    OCR lines to correct:
    {lines_text}

    Corrected lines:
    """

        try:
            model_name = getattr(self.config, 'ollama_model', 'qwen3:32b')
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.config.llm_timeout_seconds
            )

            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()

                # Parse numbered lines from response
                corrected_map = {}
                for line in raw_response.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'^(\d+)\.\s*(.*)', line)
                    if match:
                        idx = int(match.group(1)) - 1
                        text = match.group(2).strip()
                        if 0 <= idx < len(non_empty):
                            corrected_map[idx] = text

                # Apply corrections back to all segments
                non_empty_index = 0
                corrected_segments = []

                for segment in line_segments:
                    corrected_segment = segment.copy()

                    if segment.get('ocr_text', '').strip():
                        corrected_text = corrected_map.get(
                            non_empty_index,
                            segment.get('ocr_text', '')
                        )
                        corrected_segment['ocr_text_corrected'] = corrected_text
                        corrected_segment['correction_status'] = 'success'
                        non_empty_index += 1
                    else:
                        corrected_segment['ocr_text_corrected'] = ''
                        corrected_segment['correction_status'] = 'empty'

                    corrected_segments.append(corrected_segment)

                logger.info(
                    f"Batch LLM correction complete: "
                    f"{len(corrected_map)}/{len(non_empty)} lines corrected "
                    f"using {model_name}"
                )
                return corrected_segments

        except Exception as e:
            logger.error(f"Batch LLM correction failed: {e}")

        # Fallback if LLM call fails — return originals unchanged
        for segment in line_segments:
            segment['ocr_text_corrected'] = segment.get('ocr_text', '')
            segment['correction_status'] = 'llm_failed'
        return line_segments
    
    def manual_fallback_test(self) -> Tuple[str, str]:
        """Test function to verify fallback system"""
        test_text = "Prueba: texto con errores OCR para verificar el sistema de fallback..."
        logger.info("=== Manual Fallback Test ===")
        logger.info(f"Input: {test_text}")
        
        result, status = self.process_text_with_fallbacks(test_text)
        
        logger.info(f"Output: {result}")
        logger.info(f"Backend used: {status}")
        logger.info("=== Test Complete ===")
        
        return result, status
