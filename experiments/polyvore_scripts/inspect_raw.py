import json

with open("data/Polyvore/polyvore_item_metadata.json", "r") as f:
    metadata = json.load(f)

# Print the actual keys and fields available for the first item
first_id = list(metadata.keys())[0]
print(f"Raw metadata keys for {first_id}:", metadata[first_id])