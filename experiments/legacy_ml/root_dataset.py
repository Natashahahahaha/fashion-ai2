import os
import torch
from torch.utils.data import Dataset

class PolyvorePairDataset(Dataset):
    def __init__(self, pairs, labels, embeddings_dir):
        """
        pairs: List of tuples [(item_id_a, item_id_b), ...]
        labels: List of floats [1.0 (match), 0.0 (clash), ...]
        embeddings_dir: Path to the folder containing the extracted .pt files
        """
        self.pairs = pairs
        self.labels = labels
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_a_id, item_b_id = self.pairs[idx]
        
        # Load the pre-computed CLIP vectors safely
        path_a = os.path.join(self.embeddings_dir, f"{item_a_id}.pt")
        path_b = os.path.join(self.embeddings_dir, f"{item_b_id}.pt")
        
        emb_a = torch.load(path_a, weights_only=True)
        emb_b = torch.load(path_b, weights_only=True)
        
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return emb_a, emb_b, label