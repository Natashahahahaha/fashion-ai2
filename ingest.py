import sys
from src.ingestion.detector import FashionDetector
from src.wardrobe.manager import WardrobeManager

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <image_path>")
        sys.exit(1)
        
    target_image = sys.argv[1]
    
    # 1. Detect
    detector = FashionDetector()
    print(f"\nDetecting items in {target_image}...")
    detections = detector.detect(target_image, confidence_threshold=0.6)
    
    # 2. Crop, Embed, and Store
    manager = WardrobeManager()
    print("\nCropping, generating visual embeddings, and saving to Wardrobe DB...")
    added = manager.process_and_add(target_image, detections)
    
    print(f"\nSUCCESS! Added {len(added)} items to your digital wardrobe:")
    for item in added:
        print(f" -> {item['label']} (ID: {item['id']})")