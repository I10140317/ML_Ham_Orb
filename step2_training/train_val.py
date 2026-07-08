import os
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from model import GatedGCNMultiTask
from parse_data import load_all_graphs, split_dataset, compute_feature_stats, FeatureNormalizedDataset

from transformers import get_cosine_schedule_with_warmup

# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Train
# =========================
def train(model, loader, optimizer, device, scaler, use_amp=True, fnode = 100.0, fedge = 1.0):
    model.train()
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0

    for data in loader:
        data = data.to(device)
        num_batch=int(data.batch.max().item()) + 1
        optimizer.zero_grad()

        with torch.amp.autocast("cuda",enabled=(device.type == "cuda" and use_amp)):
            node_pred, edge_pred = model(data)

            node_loss = F.mse_loss(node_pred, data.node_y)*num_batch/data.num_nodes
            edge_loss = F.mse_loss(edge_pred, data.edge_y)

            loss = fnode*node_loss + fedge*edge_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_node_loss += (node_loss).item()
        total_edge_loss += (edge_loss).item()

    return total_loss/len(loader), total_node_loss/len(loader), total_edge_loss/len(loader)

@torch.no_grad()
def evaluate(model, loader, device, stats):
    model.eval()
    total_loss = 0
    total_node_loss = 0
    total_node_loss_x = 0
    total_edge_loss = 0

    for data in loader:
        num_batch=int(data.batch.max().item()) + 1
        data = data.to(device)
        node_pred, edge_pred = model(data)
        node_pred_real = node_pred * stats['nodey_std'].to(device) + stats['nodey_mean'].to(device)
        edge_pred_real = edge_pred * stats['edgey_std'].to(device) + stats['edgey_mean'].to(device)
        node_y_real = data.node_y * stats['nodey_std'].to(device) + stats['nodey_mean'].to(device)
        edge_y_real = data.edge_y * stats['edgey_std'].to(device) + stats['edgey_mean'].to(device)
        l1_loss = nn.L1Loss(reduction='mean')
        node_loss = l1_loss(node_pred_real, node_y_real)
        node_loss_x = l1_loss(node_pred_real, node_y_real)*num_batch/data.num_nodes
        edge_loss = l1_loss(edge_pred_real, edge_y_real)

        total_loss += (node_loss_x + edge_loss).item()
        total_node_loss += (node_loss).item()
        total_node_loss_x += (node_loss_x).item()
        total_edge_loss += (edge_loss).item()

    return total_loss/len(loader), total_node_loss/len(loader), total_node_loss_x/len(loader), total_edge_loss/len(loader)

# =========================
# Main
# =========================
def main(input_dirs, batch_size = 16, lr = 1e-3, epochs = 6000, use_amp = True, seed=42):

    set_seed(seed)
    device = get_device()
    print("Device:", device)

    dataset = []
    for input_dir in input_dirs:
        dataset_per = load_all_graphs(input_dir, predict=False)
        dataset.extend(dataset_per)

    print(len(dataset))
    train_set, val_set, test_set = split_dataset(dataset)
    train_set = train_set
    stats = compute_feature_stats(train_set)
    train_dataset_norm = FeatureNormalizedDataset(train_set, stats)
    val_dataset_norm = FeatureNormalizedDataset(val_set, stats)
    test_dataset_norm = FeatureNormalizedDataset(test_set, stats)

    print(stats)
    torch.save(stats,"stats.pt")

    train_loader = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_norm, batch_size=batch_size)
    test_loader = DataLoader(test_dataset_norm, batch_size=batch_size)

    model = GatedGCNMultiTask(node_input_dim = 3, edge_input_dim = 3, hidden_dim =  64, out_dim = 1, num_layers = 4, dropout = 0.1).to(device)
    #model.load_state_dict(torch.load("best_model.pt", map_location=device))
    scaler = torch.amp.GradScaler( "cuda",enabled=(device.type == "cuda" and use_amp))

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode="min",
    #    factor=0.5,
    #    patience=200,
    #    threshold=1e-4,
    #    min_lr=1e-6,
    #)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps= int(0.1*epochs),
    num_training_steps= epochs)

    best_val = 1e9

    with open("train.log", "w") as logf:
        for epoch in range(epochs):
            train_loss, train_node_loss, train_edge_loss = train(model, train_loader, optimizer, device, scaler, fnode = 100.0, fedge = 1.0)
            val_loss, val_node_loss, val_node_loss_x, val_edge_loss = evaluate(model, val_loader, device, stats)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            msg=f"Epoch {epoch+1} | Train {train_loss:.8e} | Train-node {train_node_loss:.8e} | Train-edge {train_edge_loss:.8e} | Val {val_loss:.8e} | Val-node {val_node_loss:.8e} | Val-node_x {val_node_loss_x:.8e} | Val-edge {val_edge_loss:.8e} | LR {current_lr:.2e}"
            print(msg)
            logf.write(msg + "\n")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "best_model.pt")

    print("\nTesting...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_loss, test_node_loss, test_node_loss_x, test_edge_loss = evaluate(model, test_loader, device, stats)
    print("Test Loss:", test_loss)

if __name__ == "__main__":
    prefix="../step1_data_generation"
    suffix="/output_data"
    input_dirs = []
    for i in range(1):
        #input_dir=prefix+str(i*6000)+"_"+str((i+1)*6000)+suffix
        input_dir=prefix+suffix
        input_dirs.append(input_dir)
    print(input_dirs)

    main(input_dirs, batch_size = 16, lr = 1e-3, epochs = 6000, use_amp = True, seed=42)


