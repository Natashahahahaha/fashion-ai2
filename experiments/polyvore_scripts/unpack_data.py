import json
import os
import urllib.request
from datasets import load_dataset

# Ensure directories exist
os.makedirs('data/Polyvore/images', exist_ok=True)
os.makedirs('data/Polyvore/nondisjoint', exist_ok=True)

print("Loading Parquet dataset...")
ds = load_dataset('owj0421/polyvore', split='data')

metadata = {}
total = len(ds)
print(f"Extracting {total} images and building metadata. This will take a few minutes...")

for i, row in enumerate(ds):
    item_id = str(row['item_id'])
    img_path = f"data/Polyvore/images/{item_id}.jpg"
    
    # Dump image to disk
    if not os.path.exists(img_path):
        row['image'].convert('RGB').save(img_path)
        
    # Format metadata
    metadata[item_id] = {
        "semantic_category": row['category']
    }
    
    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1} / {total} items...")

with open('data/Polyvore/polyvore_item_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Images and metadata successfully extracted.")

# Download missing outfit pairings
print("\nFetching missing nondisjoint outfit JSONs from the original repository...")
repo_url = "https://raw.githubusercontent.com/mvasil/fashion-compatibility/master/data/polyvore_outfits/nondisjoint/"

for filename in ['train.json', 'valid.json', 'test.json']:
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(f"{repo_url}{filename}", f"data/Polyvore/nondisjoint/{filename}")
    except Exception as e:
        print(f"Error fetching {filename}: {e}")

print("\nData preparation complete!")