import os
import random
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from model import GatedGCNMultiTask
from parse_data import load_all_graphs, split_dataset, compute_feature_stats, FeatureNormalizedDataset

# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_symmetric_matrix(node_y, edge_y, edge_index, num_nodes):
    device = node_y.device

    M = torch.zeros(num_nodes, num_nodes, device=device)

    idx = torch.arange(num_nodes, device=device)
    M[idx, idx] = node_y.squeeze(-1)

    src, dst = edge_index
    M[src, dst] = edge_y.squeeze(-1)
    M[dst, src] = edge_y.squeeze(-1)

    return M

@torch.no_grad()
def predict(model, loader, device, stats):
    model.eval()

    total_epsilon_loss = 0
    total_overlap_loss = 0

    for data in loader:
        data = data.to(device)
        node_pred, edge_pred = model(data)
        node_pred_real = node_pred * stats['nodey_std'].to(device) + stats['nodey_mean'].to(device)
        edge_pred_real = edge_pred * stats['edgey_std'].to(device) + stats['edgey_mean'].to(device)
        H_pred = build_symmetric_matrix(node_pred_real,edge_pred_real,data.edge_index,data.num_nodes)
        H_real = data.fock_hyb_saiao
        epsilon_pred, C_pred = torch.linalg.eigh(H_pred)
        epsilon_real, C_real = torch.linalg.eigh(H_real)

        #epsilon_ori = np.load("mo_energy_hyb.npy")
        #epsilon_ori = torch.from_numpy(epsilon_ori).to(device)

        l1_loss = nn.L1Loss(reduction='mean')
        I_pred = C_pred.T @ C_real
        overlap_loss = l1_loss(torch.diag(torch.abs(I_pred)), torch.diag(torch.eye(C_pred.shape[0]).to(device)))
        epsilon_loss = l1_loss(epsilon_pred, epsilon_real)

        total_epsilon_loss += epsilon_loss.item()
        total_overlap_loss += overlap_loss.item()
    return total_epsilon_loss/len(loader), total_overlap_loss/len(loader)

# =========================
# Main
# =========================
def main(input_dirs, batch_size = 1, seed=42, predict_bool = True):
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    dataset = []
    for input_dir in input_dirs:
        dataset_per = load_all_graphs(input_dir, predict_bool)
        dataset.extend(dataset_per)

    print(len(dataset))
    train_set, val_set, test_set = split_dataset(dataset)
    #stats = compute_feature_stats(train_set)
    stats = torch.load("stats.pt")
    #train_dataset_norm = FeatureNormalizedDataset(train_set, stats)
    #val_dataset_norm = FeatureNormalizedDataset(val_set, stats)
    test_dataset_norm = FeatureNormalizedDataset(test_set, stats)

    print(stats)
    torch.save(stats,"stats.pt")

    #train_loader = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset_norm, batch_size=batch_size)
    test_loader = DataLoader(test_dataset_norm, batch_size=batch_size)

    model = GatedGCNMultiTask(node_input_dim = 3, edge_input_dim = 3, hidden_dim =  64, out_dim = 1, num_layers = 4, dropout = 0.1).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    epsilon_loss, overlap_loss = predict(model,test_loader,device,stats)

    print(
        f"MO Energy MAE  (a.u.) {epsilon_loss:.8e} | "
        f"MO Overlap MAE {overlap_loss:.8e}"
    )

if __name__ == "__main__":
    prefix="../step1_data_generation"
    suffix="/output_data"
    input_dirs = []
    for i in range(1):
        #input_dir=prefix+str(i*6000)+"_"+str((i+1)*6000)+suffix
        input_dir=prefix+suffix
        input_dirs.append(input_dir)
    print(input_dirs)

    main(input_dirs)
