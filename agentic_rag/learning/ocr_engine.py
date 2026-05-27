"""OCR engine using GOT-OCR2.0 (stepfun-ai/GOT-OCR-2.0-hf) with graceful fallback.

This module wraps the HuggingFace GOT-OCR2.0 model for high-quality document
and image text extraction. If the model cannot be loaded (no GPU, missing deps),
it falls back to a basic image-description placeholder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..rag.utils_logger import get_logger

logger = get_logger(__name__)

# Configurable via env
OCR_MODEL_ID = os.getenv("OCR_MODEL", "stepfun-ai/GOT-OCR-2.0-hf")


class OCREngine:
    """Wraps GOT-OCR2.0 for document/image text extraction."""

    def __init__(self, model_id: str = OCR_MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading OCR model %s on %s ...", self.model_id, self._device)

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map=self._device,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            self._available = True
            logger.info("OCR model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Could not load OCR model '%s': %s.  "
                "Image OCR will use fallback (no text extraction from images).",
                self.model_id, exc,
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_text(self, image_path: str) -> str:
        """Extract text from an image file using GOT-OCR2.0."""
        if not self._available:
            return self._fallback_extract(image_path)

        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self._processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=4096,
                )

            text = self._processor.decode(
                generated_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return text.strip()
        except Exception as exc:
            logger.error("OCR extraction failed for %s: %s", image_path, exc)
            return self._fallback_extract(image_path)

    @staticmethod
    def _fallback_extract(image_path: str) -> str:
        """Minimal fallback when OCR model is not available."""
        return f"[OCR model unavailable — could not extract text from image: {Path(image_path).name}]"


# ── Singleton ─────────────────────────────────────────────────────

_engine: Optional[OCREngine] = None


def get_ocr_engine() -> OCREngine:
    """Return a lazily-initialised singleton OCR engine."""
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine
