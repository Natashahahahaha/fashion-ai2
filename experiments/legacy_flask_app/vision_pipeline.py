from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from PIL import Image

# GPU is used by the PyTorch/Ultralytics/SAM stages when available.
try:
    import torch
except Exception:
    torch = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads" / "wardrobe"
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"
YOLO_PATHS = [
    BASE_DIR / "models" / "yolov8s-world.pt",
    PROJECT_DIR / "yolov8s-world.pt",
    PROJECT_DIR / "models" / "yolov8s-world.pt",
]
SAM_PATHS = [
    BASE_DIR / "models" / "sam_vit_b_01ec64.pth",
    PROJECT_DIR / "sam_vit_b_01ec64.pth",
]
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
VISION_MODEL = "llava:latest"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        value = json.loads(text[a:b + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("Vision model did not return valid JSON")


def _list(value: Any, limit: int) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []


def _normalize_attributes(raw: dict[str, Any], detected_category: str | None = None) -> dict[str, Any]:
    formality = str(raw.get("formality", "unknown")).strip().lower()
    if formality not in {"casual", "smart-casual", "formal", "unknown"}:
        formality = "unknown"
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    category = str(raw.get("category", "unknown")).strip().lower() or "unknown"
    if category == "unknown" and detected_category:
        category = detected_category

    return {
        "category": category,
        "garment_type": str(raw.get("garment_type", "unknown")).strip() or "unknown",
        "colors": _list(raw.get("colors"), 6),
        "pattern": str(raw.get("pattern", "unknown")).strip() or "unknown",
        "material": str(raw.get("material", "unknown")).strip() or "unknown",
        "fit": str(raw.get("fit", "unknown")).strip() or "unknown",
        "style_tags": _list(raw.get("style_tags"), 6),
        "formality": formality,
        "season": _list(raw.get("season"), 4),
        "occasion": _list(raw.get("occasion"), 5),
        "confidence": max(0.0, min(1.0, confidence)),
    }


class VisionPipeline:
    """Upload -> detection -> segmentation -> VLM attributes.

    Detection and segmentation are optional at runtime so the upload path can
    still function if one vision dependency/checkpoint is unavailable. The
    pipeline never fabricates a detection: missing stages are explicitly
    recorded in the result.
    """

    def __init__(self) -> None:
        # Heavy vision models are loaded lazily and released between stages.
        # This is important on a 16-GB RAM / 8-GB VRAM machine because Ollama/LLaVA
        # also needs several GB of system memory when it loads the vision model.
        self.yolo = None
        self.sam = None
        self.sam_predictor = None
        self.yolo_path = _first_existing(YOLO_PATHS)
        self.sam_path = _first_existing(SAM_PATHS)
        print(
            f"[vision] pipeline ready | device={DEVICE} | "
            f"YOLO={'yes' if self.yolo_path else 'no'} | SAM={'yes' if self.sam_path else 'no'}"
        )

    def _release_gpu_model(self, attr_name: str) -> None:
        obj = getattr(self, attr_name, None)
        if obj is not None:
            try:
                del obj
            except Exception:
                pass
            setattr(self, attr_name, None)

        if attr_name == "sam":
            self.sam_predictor = None

        import gc
        gc.collect()

        if torch is not None and DEVICE == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_yolo(self) -> bool:
        if self.yolo is not None:
            return True
        if self.yolo_path is None:
            print("[vision] YOLO checkpoint not found; continuing without YOLO.")
            return False

        try:
            from ultralytics import YOLO
            self.yolo = YOLO(str(self.yolo_path))
            if hasattr(self.yolo, "to"):
                self.yolo.to(DEVICE)
            self.yolo.set_classes([
                "shirt", "t-shirt", "blouse", "top", "tank top",
                "pants", "trousers", "jeans", "skirt", "shorts",
                "dress", "jumpsuit", "jacket", "coat", "sweater",
                "hoodie", "shoes", "sneakers", "sandals", "boots",
                "bag", "handbag", "backpack", "jewelry", "necklace",
                "earrings", "bracelet", "hat", "scarf",
            ])
            print(f"[vision] YOLO-World loaded | {self.yolo_path} | device={DEVICE}")
            return True
        except Exception as exc:
            print(f"[vision] YOLO unavailable: {type(exc).__name__}: {exc}")
            self.yolo = None
            return False

    def _load_sam(self) -> bool:
        if self.sam_predictor is not None:
            return True
        if self.sam_path is None:
            print("[vision] SAM checkpoint not found; continuing without segmentation.")
            return False

        try:
            from segment_anything import SamPredictor, sam_model_registry
            self.sam = sam_model_registry["vit_b"](checkpoint=str(self.sam_path))
            self.sam.to(device=DEVICE)
            self.sam_predictor = SamPredictor(self.sam)
            print(f"[vision] SAM loaded | {self.sam_path} | device={DEVICE}")
            return True
        except Exception as exc:
            print(f"[vision] SAM unavailable: {type(exc).__name__}: {exc}")
            self.sam = None
            self.sam_predictor = None
            return False


    def detect(self, image_path: Path) -> dict[str, Any]:
        if not self._load_yolo():
            return {"available": False, "detections": []}

        try:
            results = self.yolo.predict(
                source=str(image_path),
                device=0 if DEVICE == "cuda" else "cpu",
                conf=0.20,
                verbose=False,
            )
            result = results[0]
            detections = []
            names = result.names
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                xyxy = [round(float(v), 2) for v in boxes.xyxy[i].tolist()]
                detections.append({
                    "label": str(names[cls_id]).lower(),
                    "confidence": round(conf, 4),
                    "box": xyxy,
                })
            detections.sort(key=lambda x: -x["confidence"])
            return {"available": True, "detections": detections[:10]}
        except Exception as exc:
            return {"available": False, "detections": [], "error": str(exc)}
        finally:
            # Free YOLO before SAM / Ollama is used.
            self._release_gpu_model("yolo")

    def segment_best(self, image_path: Path, detections: list[dict[str, Any]]) -> Path | None:
        if not detections or not self._load_sam():
            return None

        try:
            import numpy as np
            image = np.array(Image.open(image_path).convert("RGB"))
            self.sam_predictor.set_image(image)
            box = np.array(detections[0]["box"], dtype=np.float32)
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best]

            rgba = np.dstack([image, (mask.astype("uint8") * 255)])
            out = PROCESSED_DIR / f"{image_path.stem}_cutout.png"
            Image.fromarray(rgba).save(out)
            return out
        except Exception as exc:
            print(f"[vision] SAM segmentation failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            # Free SAM before the LLaVA request.
            self._release_gpu_model("sam")

    def attributes(self, image_path: Path, detected_category: str | None = None) -> dict[str, Any]:
        prompt = f"""You are the structured fashion-attribute stage of a wardrobe digitization system.
Analyze ONLY the clothing/accessory item in this image.
A detector may have suggested category: {detected_category or 'unknown'}.

Return ONLY valid JSON with exactly these fields:
{{
  "category": "tops | bottoms | all-body | outerwear | shoes | bags | jewellery | accessories | hats | scarves | unknown",
  "garment_type": "specific visible item type or unknown",
  "colors": ["dominant visible colors"],
  "pattern": "solid | floral | striped | plaid | checked | graphic | paisley | embroidered | other | unknown",
  "material": "visually plausible material or unknown",
  "fit": "visible silhouette/cut or unknown",
  "style_tags": ["2-6 grounded visual descriptors"],
  "formality": "casual | smart-casual | formal | unknown",
  "season": ["spring", "summer", "autumn", "winter"],
  "occasion": ["2-5 plausible occasions"],
  "confidence": 0.0
}}

Rules:
- Do not invent hidden details, brand, price, or identity.
- If material or fit is not visually reliable, use unknown.
- Ground style tags in visible design.
- The image may be a real wardrobe photo, not a catalog image.
"""
        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [_b64(image_path)]}],
            "options": {"temperature": 0},
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=240)
        if not response.ok:
            detail = response.text.strip()
            raise RuntimeError(
                f"Ollama/LLaVA HTTP {response.status_code}: "
                f"{detail[:1200] if detail else 'no response body'}"
            )
        raw = response.json()["message"]["content"]
        return _normalize_attributes(_parse_json(raw), detected_category)

    def process(self, image_path: Path) -> dict[str, Any]:
        detection = self.detect(image_path)
        best_detection = detection.get("detections", [None])[0]
        detected_category = None
        if best_detection:
            detected_category = self._map_category(best_detection["label"])

        cutout = self.segment_best(image_path, detection.get("detections", []))
        analysis_image = cutout if cutout is not None else image_path

        # At this point YOLO/SAM have been released, so Ollama/LLaVA gets
        # maximum available system RAM and VRAM.
        attrs = self.attributes(analysis_image, detected_category)

        return {
            "vision": {
                "device": DEVICE,
                "detector": "YOLO-World" if detection.get("available") else None,
                "detections": detection.get("detections", []),
                "segmented": cutout is not None,
                "analysis_image": str(analysis_image.relative_to(BASE_DIR)).replace("\\", "/"),
            },
            "attributes": attrs,
        }

    @staticmethod
    def _map_category(label: str) -> str:
        label = label.lower()
        if label in {"shirt", "t-shirt", "blouse", "top", "tank top", "hoodie"}:
            return "tops"
        if label in {"pants", "trousers", "jeans", "skirt", "shorts"}:
            return "bottoms"
        if label in {"dress", "jumpsuit"}:
            return "all-body"
        if label in {"jacket", "coat", "sweater"}:
            return "outerwear"
        if label in {"shoes", "sneakers", "sandals", "boots"}:
            return "shoes"
        if label in {"bag", "handbag", "backpack"}:
            return "bags"
        if label in {"jewelry", "necklace", "earrings", "bracelet"}:
            return "jewellery"
        if label in {"hat"}:
            return "hats"
        if label in {"scarf"}:
            return "scarves"
        return "accessories"
