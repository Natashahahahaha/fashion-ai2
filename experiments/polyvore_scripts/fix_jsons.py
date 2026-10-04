import urllib.request
import os

os.makedirs('data/Polyvore/nondisjoint', exist_ok=True)
base_url = "https://huggingface.co/datasets/owj0421/polyvore-outfits/resolve/main/nondisjoint/"

for filename in ['train.json', 'valid.json', 'test.json']:
    print(f"Fetching {filename} from Hugging Face mirror...")
    try:
        urllib.request.urlretrieve(f"{base_url}{filename}", f"data/Polyvore/nondisjoint/{filename}")
        print(f"Success: {filename} downloaded.")
    except Exception as e:
        print(f"Error fetching {filename}: {e}")