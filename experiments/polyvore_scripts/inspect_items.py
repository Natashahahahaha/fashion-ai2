import json

with open("data/Polyvore/polyvore_item_metadata.json", "r") as f:
    metadata = json.load(f)

target_ids = ["211990161", "183179503", "182043322", "120679156"]

for item_id in target_ids:
    if item_id in metadata:
        info = metadata[item_id]
        print(f"ID: {item_id} | Semantic Category: {info.get('semantic_category', 'Unknown')}")
    else:
        print(f"ID: {item_id} | Not found in metadata")