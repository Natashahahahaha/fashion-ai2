import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.ml.model import OutfitCompatibilityNet
import os

def generate_synthetic_fashion_dataset(n_samples=2400, emb_dim=512):
    torch.manual_seed(42)
    n_pairs = n_samples // 2
    
    # Positives: Vector pairs with coherent structural relationships
    base_style = torch.randn(n_pairs, emb_dim)
    noise_pos = torch.randn(n_pairs, emb_dim) * 0.35
    top_pos = F_norm(base_style + noise_pos)
    bottom_pos = F_norm(base_style - noise_pos)
    labels_pos = torch.ones(n_pairs, 1)

    # Negatives: Clashing vectors
    top_neg = F_norm(torch.randn(n_pairs, emb_dim))
    bottom_neg = F_norm(torch.randn(n_pairs, emb_dim))
    labels_neg = torch.zeros(n_pairs, 1)

    top_all = torch.cat([top_pos, top_neg], dim=0)
    bottom_all = torch.cat([bottom_pos, bottom_neg], dim=0)
    labels_all = torch.cat([labels_pos, labels_neg], dim=0)

    indices = torch.randperm(n_samples)
    return top_all[indices], bottom_all[indices], labels_all[indices]

def F_norm(tensor):
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing PyTorch Training Engine on: {device.upper()}")

    top, bottom, labels = generate_synthetic_fashion_dataset(n_samples=3000)
    
    split = int(0.8 * len(labels))
    train_ds = TensorDataset(top[:split], bottom[:split], labels[:split])
    test_ds = TensorDataset(top[split:], bottom[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = OutfitCompatibilityNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    epochs = 15
    print("\n--- INITIATING PYTORCH MODEL TRAINING ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_top, b_bottom, b_labels in train_loader:
            b_top, b_bottom, b_labels = b_top.to(device), b_bottom.to(device), b_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(b_top, b_bottom)
            loss = criterion(predictions, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_labels)
            
        epoch_loss = total_loss / len(train_loader.dataset)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

    model.eval()
    test_top, test_bottom, test_labels = top[split:].to(device), bottom[split:].to(device), labels[split:].numpy()
    
    with torch.no_grad():
        ml_preds_prob = model(test_top, test_bottom).cpu().numpy()
        ml_preds_binary = (ml_preds_prob >= 0.5).astype(int)

        cosine_sim = torch.sum(test_top * test_bottom, dim=-1).cpu().numpy()
        baseline_preds_binary = (cosine_sim >= cosine_sim.mean()).astype(int)

    ml_acc = accuracy_score(test_labels, ml_preds_binary)
    ml_f1 = f1_score(test_labels, ml_preds_binary)
    ml_auc = roc_auc_score(test_labels, ml_preds_prob)
    base_acc = accuracy_score(test_labels, baseline_preds_binary)
    base_f1 = f1_score(test_labels, baseline_preds_binary)

    print("\n================ BENCHMARK RESULTS ================")
    print(f"BASELINE (Cosine Similarity): Accuracy = {base_acc*100:.2f}%, F1 = {base_f1:.4f}")
    print(f"PROPOSED (OutfitCompatibilityNet): Accuracy = {ml_acc*100:.2f}%, F1 = {ml_f1:.4f}, AUC = {ml_auc:.4f}")
    print("====================================================")

    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/compatibility_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTrained weights saved to: {save_path}")

if __name__ == "__main__":
    main()