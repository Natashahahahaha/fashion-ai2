import os
import torch
import torch.nn.functional as F
from src.ml.model import OutfitCompatibilityNet

class CompatibilityEngine:
    def __init__(self, model_path="data/processed/compatibility_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = OutfitCompatibilityNet().to(self.device)
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded trained compatibility weights from {model_path}")
        else:
            print("No pretrained weights found. Running in baseline/untrained state.")

    def baseline_score(self, emb1, emb2):
        """Baseline approach: Pure normalized cosine similarity."""
        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        return float((sim.item() + 1.0) / 2.0)

    def ml_score(self, emb1, emb2):
        """Learned approach: Inference via OutfitCompatibilityNet."""
        self.model.eval()
        with torch.no_grad():
            score = self.model(emb1.to(self.device), emb2.to(self.device))
        return float(score.item())

    def compare(self, emb1_path, emb2_path):
        emb1 = torch.load(emb1_path).squeeze(0)
        emb2 = torch.load(emb2_path).squeeze(0)
        
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        base = self.baseline_score(emb1, emb2)
        ml = self.ml_score(emb1, emb2)
        
        return {
            "baseline_cosine_score": round(base, 4),
            "learned_ml_score": round(ml, 4)
        }