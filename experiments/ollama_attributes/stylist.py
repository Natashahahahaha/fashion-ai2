import os
import sys
import ollama

def classify_wardrobe_items(crop_dir):
    if not os.path.exists(crop_dir):
        print(f"Directory not found: {crop_dir}")
        sys.exit(1)
        
    print(f"Waking up LLaVA to analyze crops in {crop_dir}...\n")
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    for filename in os.listdir(crop_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue
            
        image_path = os.path.join(crop_dir, filename)
        print(f"--- Analyzing: {filename} ---")
        
        # Chain-of-Thought prompt forcing visual grounding before classification
        prompt = (
            "You are an expert fashion archivist. Analyze this single clothing item carefully. "
            "Do not use generic labels. Look at the specific cut, rise, hem, material, and hardware.\n\n"
            "Output exactly three lines:\n"
            "1. Details: [Briefly state the defining physical features, e.g., low-rise waist with flared hem, velour texture with front pockets]\n"
            "2. Specific Cut: [Exact sub-type based on details, e.g., low-rise bootcut jeans, velour track hoodie, slouchy motorcycle satchel]\n"
            "3. Aesthetic: [Exact historical/core aesthetic, e.g., Y2K, 2000s Mall Goth, Grunge, Streetwear]\n"
        )
        
        try:
            # Send the image directly to local LLaVA
            response = ollama.generate(
                model='llava',
                prompt=prompt,
                images=[image_path]
            )
            print(response['response'].strip() + "\n")
            
        except Exception as e:
            print(f"Inference failed. Is Ollama running? Error: {e}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/agent/stylist.py <path_to_cropped_folder>")
        sys.exit(1)
        
    classify_wardrobe_items(sys.argv[1])