from transformers import pipeline
from typing import Dict, Any, List

class LocalGenerator:
    def __init__(self, model_name: str):
        self.pipe = pipeline('text2text-generation', model=model_name)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        text = out[0].get('generated_text') or out[0].get('summary_text') or ''
        return text.strip()
