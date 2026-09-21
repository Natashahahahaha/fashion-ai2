import os
import json
import uuid
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class WardrobeManager:
    def __init__(self, storage_dir="data/processed"):
        self.storage_dir = storage_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.registry_file = os.path.join(storage_dir, "wardrobe_registry.json")
        self.embeddings_dir = os.path.join(storage_dir, "embeddings")

        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

        print(f"Loading CLIP model on {self.device.upper()} for visual embeddings...")
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)

    def process_and_add(self, original_image_path, detections):
        if not detections:
            return []

        original_img = Image.open(original_image_path).convert("RGB")
        added_items = []

        for det in detections:
            item_id = str(uuid.uuid4())[:8]
            label = det["item"]
            box = det["box"]  # [xmin, ymin, xmax, ymax]

            # 1. Crop image
            # Ensure box is within image boundaries to prevent errors
            w, h = original_img.size
            box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]

            # Skip invalid boxes
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            cropped_img = original_img.crop((box[0], box[1], box[2], box[3]))
            safe_label = label.replace(" ", "_").replace("/", "_")
            crop_path = os.path.join(self.storage_dir, f"{item_id}_{safe_label}.jpg")
            cropped_img.save(crop_path)

            # 2. Generate Embedding
            inputs = self.processor(images=cropped_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the embeddings for cosine similarity later
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # Save embedding tensor
            emb_path = os.path.join(self.embeddings_dir, f"{item_id}.pt")
            torch.save(image_features.cpu(), emb_path)

            # 3. Update Registry
            metadata = {
                "id": item_id,
                "label": label,
                "confidence": det["confidence"],
                "image_path": crop_path,
                "embedding_path": emb_path,
                "source_image": original_image_path
            }
            self.registry[item_id] = metadata
            added_items.append(metadata)

        # Save registry
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=4)

        return added_items