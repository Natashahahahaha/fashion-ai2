"""
Turns a single cropped clothing-item image into a structured WardrobeItem
by prompting a LOCAL vision model via Ollama — no API key, no cost.

Setup (one-time):
    ollama pull llava
    # keep this running in a separate terminal window while you test:
    ollama serve

If you'd rather use Llama 3.2 Vision instead (better quality, needs more
VRAM/disk — 11B, ~8GB VRAM, vs llava's ~4.7GB download / lighter to run):
    ollama pull llama3.2-vision
    # then pass model="llama3.2-vision" to AttributeExtractor below

Same honesty note as before: this is a prompted foundation model doing
zero-shot attribute extraction, not a custom-trained classifier — say
that plainly if asked, it's a legitimate v1 choice given zero labeled
data, whether the model is hosted (Claude/GPT-4V) or local (Ollama).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ollama_client import call_ollama_chat, is_ollama_available
from schema import WardrobeItem, Formality

ATTRIBUTE_SCHEMA_PROMPT = """You are extracting structured attributes from a single cropped photo of ONE clothing item.

Return ONLY a JSON object (no prose, no markdown fences, no explanation) with exactly these keys:

{
  "is_clothing_item": true or false,
  "category": string,
  "subcategory": string or null,
  "color": string or null,
  "pattern": string or null,
  "silhouette": string or null,
  "sleeve": string or null,
  "neckline": string or null,
  "material": string or "uncertain",
  "formality": one of ["casual","smart_casual","formal","athletic","unknown"],
  "season": string or null,
  "style_tags": array of short strings, e.g. ["streetwear","minimal"],
  "confidence": float between 0 and 1
}

Rules:
- If you are not confident about material, return "uncertain". Never guess a specific fabric from a photo alone.
- If the crop is not a wearable clothing/accessory item (background, furniture, skin, an unusable fragment), set "is_clothing_item": false and leave the other fields null.
- confidence should reflect genuine certainty about category + color specifically, not the whole object.
- Return ONLY the JSON object. Nothing before it, nothing after it.
"""


class AttributeExtractor:
    def __init__(self, model: str = "llava"):
        self.model = model
        if not is_ollama_available():
            print(
                "[AttributeExtractor] WARNING: Ollama doesn't seem to be running at "
                "localhost:11434. Run `ollama serve` in another terminal, and make "
                f"sure you've pulled the model with `ollama pull {model}`."
            )

    def extract(self, crop_path: str) -> Optional[WardrobeItem]:
        try:
            raw_text = call_ollama_chat(self.model, ATTRIBUTE_SCHEMA_PROMPT, image_path=crop_path)
        except RuntimeError as e:
            print(f"[AttributeExtractor] {e}")
            return None

        text = raw_text.strip()
        # local models are chattier about wrapping JSON in fences even when told not to
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # some vision models add a sentence before/after despite instructions — grab the {...} span
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"[AttributeExtractor] Could not parse model output for {crop_path}: {text[:200]}")
            return None

        if not data.get("is_clothing_item", False):
            return None

        confidence = float(data.get("confidence", 0.0))

        try:
            formality = Formality(data.get("formality", "unknown"))
        except ValueError:
            formality = Formality.UNKNOWN

        return WardrobeItem(
            item_id=os.path.splitext(os.path.basename(crop_path))[0],
            category=data.get("category", "unknown"),
            subcategory=data.get("subcategory"),
            color=data.get("color"),
            pattern=data.get("pattern"),
            silhouette=data.get("silhouette"),
            sleeve=data.get("sleeve"),
            neckline=data.get("neckline"),
            material=data.get("material", "uncertain"),
            formality=formality,
            season=data.get("season"),
            style_tags=data.get("style_tags", []) or [],
            confidence=confidence,
            needs_review=confidence < 0.7,
            crop_path=crop_path,
        )
