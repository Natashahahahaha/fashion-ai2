from __future__ import annotations
import base64, json, os, re, time
from typing import Any
import requests

DEMO_DIR = "demo_wardrobe"
IMAGE_DIR = os.path.join(DEMO_DIR, "images")
METADATA_PATH = os.path.join(DEMO_DIR, "metadata.json")
ATTRIBUTES_PATH = os.path.join(DEMO_DIR, "attributes.json")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
VISION_MODEL = "llava:latest"

SCHEMA = {
    "garment_type": "specific visible item type or unknown",
    "colors": ["clearly visible dominant colors"],
    "pattern": "pattern, solid, or unknown",
    "material": "visually plausible material or unknown",
    "fit": "visible silhouette/cut or unknown",
    "style_tags": ["2-5 visible style descriptors"],
    "formality": "casual | smart-casual | formal | unknown",
    "season": ["spring", "summer", "autumn", "winter"],
    "occasion": ["2-4 plausible occasions"],
    "confidence": 0.0,
}

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        x = json.loads(text)
        if isinstance(x, dict): return x
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        x = json.loads(text[a:b+1])
        if isinstance(x, dict): return x
    raise ValueError("Ollama did not return parseable JSON")

def lst(x):
    if isinstance(x, list): return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str) and x.strip(): return [x.strip()]
    return []

def normalize(x, category, title):
    formality = str(x.get("formality", "unknown")).strip().lower()
    if formality not in {"casual","smart-casual","formal"}: formality = "unknown"
    try: conf = float(x.get("confidence", 0))
    except (TypeError, ValueError): conf = 0
    return {
        "title": title, "category": category,
        "garment_type": str(x.get("garment_type","unknown")).strip() or "unknown",
        "colors": lst(x.get("colors"))[:6],
        "pattern": str(x.get("pattern","unknown")).strip() or "unknown",
        "material": str(x.get("material","unknown")).strip() or "unknown",
        "fit": str(x.get("fit","unknown")).strip() or "unknown",
        "style_tags": lst(x.get("style_tags"))[:5],
        "formality": formality,
        "season": lst(x.get("season"))[:4],
        "occasion": lst(x.get("occasion"))[:4],
        "confidence": max(0, min(1, conf)),
    }

def analyze(item_id, info):
    image_path = os.path.join(IMAGE_DIR, f"{item_id}.jpg")
    if not os.path.exists(image_path): raise FileNotFoundError(image_path)
    prompt = f"""You are the computer-vision attribute extractor for a fashion wardrobe system.
Analyze ONLY the item visible in the image.
Known metadata:
- category: {info.get("category","unknown")}
- catalog title: {info.get("title","unknown")}

Return ONLY valid JSON matching this schema:
{json.dumps(SCHEMA, indent=2)}

Rules:
- Never invent details that are not visible.
- Use simple color names.
- Material = unknown when unreliable.
- Fit describes visible silhouette/cut only.
- Style tags must be grounded in visible design.
- Do not infer brand, price, identity, body type, or anything outside the item.
"""
    payload = {
        "model": VISION_MODEL, "stream": False,
        "messages": [{"role":"user","content":prompt,"images":[b64(image_path)]}],
        "options": {"temperature": 0}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return normalize(parse_json(r.json()["message"]["content"]),
                     info.get("category","unknown"), info.get("title", item_id))

def main():
    if not os.path.exists(METADATA_PATH): raise FileNotFoundError(METADATA_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f: metadata = json.load(f)
    attrs = {}
    if os.path.exists(ATTRIBUTES_PATH):
        with open(ATTRIBUTES_PATH, encoding="utf-8") as f: attrs = json.load(f)

    print(f"[vision] {len(metadata)} items | model={VISION_MODEL}")
    for n, (item_id, info) in enumerate(metadata.items(), 1):
        if item_id in attrs:
            print(f"[vision] {n}/{len(metadata)} SKIP {item_id}")
            continue
        print(f"[vision] {n}/{len(metadata)} {item_id} | {info.get('category')} | {info.get('title')}")
        try:
            attrs[item_id] = analyze(item_id, info)
            with open(ATTRIBUTES_PATH, "w", encoding="utf-8") as f:
                json.dump(attrs, f, indent=2, ensure_ascii=False)
            x = attrs[item_id]
            print(f"         -> {x['garment_type']} | {x['colors']} | conf={x['confidence']:.2f}")
        except Exception as e:
            print(f"         ERROR: {type(e).__name__}: {e}")
        time.sleep(0.25)
    print(f"[vision] DONE: {len(attrs)} records -> {ATTRIBUTES_PATH}")

if __name__ == "__main__":
    main()
