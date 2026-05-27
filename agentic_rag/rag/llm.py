"""HuggingFace local text generation wrapper with graceful fallback."""

from .utils_logger import get_logger

logger = get_logger(__name__)

SAFE_LLM_ERROR = "The AI model is temporarily unavailable. Please try again later."


class LocalGenerator:
    def __init__(self, model_name: str):
        try:
            from transformers import pipeline as hf_pipeline
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            if getattr(config, "is_encoder_decoder", False):
                task = "text2text-generation"
            else:
                task = "text-generation"

            self.pipe = hf_pipeline(task, model=model_name)
            self._available = True
            logger.info("Generator loaded: %s", model_name)
        except Exception as e:
            logger.warning("Could not load generator '%s': %s. Using echo fallback.", model_name, e)
            self.pipe = None
            self._available = False

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        if not self._available:
            return SAFE_LLM_ERROR
        try:
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-4),
            }
            # Decoder-only models (like Qwen) need return_full_text=False so they don't echo the prompt back
            if getattr(self.pipe, "task", "") == "text-generation":
                kwargs["return_full_text"] = False

            out = self.pipe(prompt, **kwargs)
            text = out[0].get("generated_text") or out[0].get("summary_text") or ""
            return text.strip()
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return SAFE_LLM_ERROR
