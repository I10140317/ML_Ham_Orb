import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

def convert_graph(data, predict = False):
    new_data = Data(
        x=data.x[:, [0,1,3]].float(),
        edge_index=data.edge_index,
        edge_attr=data.edge_attr[:, [0,1,3]].float(),
    )
    
    if predict:
        new_data.C_ao_saiao = torch.tensor(data.C_ao_saiao,dtype = torch.float32)
        new_data.Sao_saiao = torch.tensor(data.Sao_saiao, dtype = torch.float32)
        if hasattr(data, "fock_hyb_saiao"):
            new_data.fock_hyb_saiao = torch.tensor(data.fock_hyb_saiao, dtype = torch.float32)
    else:
        new_data.node_y = data.x[:, 5].float().unsqueeze(-1)
        new_data.edge_y = data.edge_attr[:, 5].float().unsqueeze(-1)

    return new_data

def load_pt_file(path, predict = False):
    """
    FIXED: weights_only=False for PyTorch 2.6
    """
    obj = torch.load(path, weights_only=False)

    if isinstance(obj, list):
        return [convert_graph(g, predict) for g in obj]
    elif isinstance(obj, Data):
        return [convert_graph(obj, predict)]
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

def load_all_graphs(input_dir, predict = False):
    """
    Load only files matching shard_XXXXXX.pt in numeric order.
    """
    pattern = re.compile(r"^shard_0000(\d+)\.pt$")

    files = []
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match is not None:
            shard_idx = int(match.group(1))
            files.append((shard_idx, fname))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No shard_*.pt files found in {input_dir}"
        )

    files = sorted(files, key=lambda x: x[0])

    dataset = []

    for shard_idx, fname in files:
        path = os.path.join(input_dir, fname)
        graphs = load_pt_file(path, predict)
        dataset.extend(graphs)
        print(f"Loaded {fname}: {len(graphs)} graphs")

    print(f"Total graphs: {len(dataset)}")
    return dataset

def split_dataset(dataset):
    random.shuffle(dataset)

    n = len(dataset)
    train = dataset[:int(0.8 * n)]
    val = dataset[int(0.8 * n):int(0.9 * n)]
    test = dataset[int(0.9 * n):]

    return train, val, test

def compute_feature_stats(dataset):
    x_list = []
    edge_list = []
    nodey_list = []
    edgey_list = []

    for data in dataset:
        if hasattr(data, "x") and data.x is not None:
            x_list.append(data.x)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_list.append(data.edge_attr)
        if hasattr(data, "node_y") and data.node_y is not None:
            nodey_list.append(data.node_y)
        if hasattr(data, "edge_y") and data.edge_y is not None:
            edgey_list.append(data.edge_y)

    stats = {}

    if len(x_list) > 0:
        x_all = torch.cat(x_list, dim=0)   # [sum_num_nodes, node_feat_dim]
        stats["x_mean"] = x_all.mean(dim=0, keepdim=True)
        stats["x_std"] = x_all.std(dim=0, keepdim=True) + 1e-12

    if len(edge_list) > 0:
        edge_all = torch.cat(edge_list, dim=0)   # [sum_num_edges, edge_feat_dim]
        stats["edge_mean"] = edge_all.mean(dim=0, keepdim=True)
        stats["edge_std"] = edge_all.std(dim=0, keepdim=True) + 1e-12

    if len(nodey_list) > 0:
        nodey_all = torch.cat(nodey_list, dim=0)   # [sum_num_nodes, node_feat_dim]
        stats["nodey_mean"] = nodey_all.mean(dim=0, keepdim=True)
        stats["nodey_std"] = nodey_all.std(dim=0, keepdim=True) + 1e-12

    if len(edgey_list) > 0:
        edgey_all = torch.cat(edgey_list, dim=0)   # [sum_num_edges, edge_feat_dim]
        stats["edgey_mean"] = edgey_all.mean(dim=0, keepdim=True)
        stats["edgey_std"] = edgey_all.std(dim=0, keepdim=True) + 1e-12

    return stats

class FeatureNormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.stats = stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()

        if hasattr(data, "x") and data.x is not None:
            data.x = (data.x - self.stats["x_mean"]) / self.stats["x_std"]

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = (data.edge_attr - self.stats["edge_mean"]) / self.stats["edge_std"]

        if hasattr(data, "node_y") and data.node_y is not None:
            data.node_y = (data.node_y - self.stats["nodey_mean"]) / self.stats["nodey_std"]

        if hasattr(data, "edge_y") and data.edge_y is not None:
            data.edge_y = (data.edge_y - self.stats["edgey_mean"]) / self.stats["edgey_std"]

        return data

