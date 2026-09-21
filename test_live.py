import json
import os
import torch
from embed import load_embedding
from model import OutfitCompatibilityNet

# 1. Load two valid item IDs from metadata
with open("data/Polyvore/polyvore_item_metadata.json", "r") as f:
    metadata = json.load(f)

item_ids = list(metadata.keys())
id_a, id_b = item_ids[0], item_ids[1]
print(f"Testing compatibility between: {id_a} and {id_b}")

# 2. Load trained model checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OutfitCompatibilityNet().to(device)
model.load_state_dict(torch.load("checkpoints/compatibility_net.pt", map_location=device, weights_only=True))
model.eval()

# 3. Load pre-computed CLIP embeddings and evaluate
emb_a = load_embedding(os.path.join("data/Polyvore/embeddings", f"{id_a}.pt")).unsqueeze(0).to(device)
emb_b = load_embedding(os.path.join("data/Polyvore/embeddings", f"{id_b}.pt")).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(emb_a, emb_b)
    score = torch.sigmoid(logits).item()

print(f"Compatibility Score: {score:.4f}")