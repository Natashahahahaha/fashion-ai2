from ultralytics import YOLOWorld
from PIL import Image
import sys
import os
import shutil

class ModernFashionDetector:
    def __init__(self):
        print("Initializing YOLO-World Vision-Language Detector...")
        self.model = YOLOWorld("yolov8s-world.pt")
        
        self.custom_classes = [
            "shirt", "t-shirt", "sweater", "cardigan", "jacket", "coat",
            "jeans", "pants", "shorts", "skirt", "dress", 
            "shoe", "sneaker", "boot", "bag", "backpack", "sunglasses"
        ]
        self.model.set_classes(self.custom_classes)

    def detect_and_crop(self, image_path, confidence_threshold=0.25):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # iou=0.4 prevents the model from generating multiple crops of the exact same item
        results = self.model.predict(image_path, conf=confidence_threshold, iou=0.4, verbose=False)
        
        base_dir = os.path.dirname(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(base_dir, f"{img_name}_cropped")
        
        # Clear out the old garbage crops if the directory exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        image = Image.open(image_path).convert("RGB")
        detected = []
        
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0].item())
            item_name = self.custom_classes[cls_id]
            conf = round(box.conf[0].item(), 3)
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = image.crop((x1, y1, x2, y2))
            
            crop_filename = f"{item_name}_{i}_{conf}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path)
            
            detected.append({
                "item": item_name,
                "confidence": conf,
                "path": crop_path
            })
            
        return detected, output_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion/detector.py <image_path>")
        sys.exit(1)
        
    detector = ModernFashionDetector()
    print("Processing image...")
    items, output_folder = detector.detect_and_crop(sys.argv[1])
    
    print("\n--- WARDROBE ITEMS DETECTED & CROPPED ---")
    if not items:
        print("No macro clothing items detected.")
    else:
        for i in items:
            print(f"- {i['item']} (Confidence: {i['confidence']}) -> Saved to: {os.path.basename(i['path'])}")
        print(f"\nAll cropped images saved to: {output_folder}")