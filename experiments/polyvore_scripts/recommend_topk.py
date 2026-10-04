import json
import os
import torch
from embed import load_embedding
from model import OutfitCompatibilityNet

with open("data/Polyvore/polyvore_item_metadata.json", "r") as f:
    metadata = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = OutfitCompatibilityNet().to(device)
model.load_state_dict(torch.load("checkpoints/compatibility_net.pt", map_location=device, weights_only=True))
model.eval()

item_ids = list(metadata.keys())
anchor_id = item_ids[0]
candidate_ids = item_ids[1:51]

print(f"Anchor Item: {anchor_id}")
print(f"Scoring {len(candidate_ids)} candidate items...")

emb_anchor = load_embedding(os.path.join("data/Polyvore/embeddings", f"{anchor_id}.pt")).unsqueeze(0).to(device)

results = []
with torch.no_grad():
    for cid in candidate_ids:
        emb_cand = load_embedding(os.path.join("data/Polyvore/embeddings", f"{cid}.pt")).unsqueeze(0).to(device)
        logits = model(emb_anchor, emb_cand)
        score = torch.sigmoid(logits).item()
        results.append((cid, score))

results.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 Compatible Recommendations:")
for cid, score in results[:5]:
    print(f"Item ID: {cid} | Compatibility Score: {score:.4f}")