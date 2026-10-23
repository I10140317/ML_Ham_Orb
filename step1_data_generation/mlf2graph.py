from pyscf import gto, dft
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import os
from pathlib import Path
import random

def build_pyg_data_from_mats(A, M_list, len_M_list, remove_self=True, undirected=True):

    A = np.asarray(A)
    N = A.shape[0]
    assert A.shape == (N, N)
    assert len(M_list) == len_M_list, "M_list must equals to the length of key lists."

    Mats = [np.asarray(M) for M in M_list]
    for k, M in enumerate(Mats):
        assert M.shape == (N, N), f"M[{k}] shape mismatch"

    x = np.stack([np.diag(M) for M in Mats], axis=1)
    x = torch.tensor(x, dtype=torch.float32)

    A_work = A.copy()
    if remove_self:
        np.fill_diagonal(A_work, 0)

    if undirected:
        iu, ju = np.triu_indices(N, k=1)
        mask = A_work[iu, ju] != 0
        i = iu[mask]
        j = ju[mask]

        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        attr_u = np.stack([M[i, j] for M in Mats], axis=1)   
        edge_attr = np.vstack([attr_u, attr_u])              
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        src, dst = np.nonzero(A_work)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_attr = np.stack([M[src, dst] for M in Mats], axis=1)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)

    return data

def batch_folder(input_dir, glob_pattern="*.pt", full_path=True):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")
    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        print(f"No files matched {glob_pattern} in {input_dir}")
        return
    if full_path:
        return [str(f.resolve()) for f in files]
    else:
        return [str(f.name) for f in files]

def data_generator_from_pt(pt_files, build_pyg_data_from_mats, 
                           key_list = ['dm_saiao','fock_saiao','hcore_saiao','vj_saiao','vxc_saiao','Sao_saiao','fock_hyb_saiao'],
                           threshold = 1e-5, remove_self = False, undirected = True, verbose = False):

    for i, pt_file in enumerate(pt_files):
        if verbose:
            print(f"[{i}] {pt_file}", flush=True)
        try:
            mlfs=torch.load(pt_file, weights_only=False)

            for j, mlf in enumerate(mlfs):
                M_list = [mlf[k] for k in key_list]
                #if 'vj_saiao' in key_list and 'vxc_saiao' in key_list: 
                #    A = ((np.abs(mlf['vj_saiao']) > threshold) | 
                #         (np.abs(mlf['vxc_saiao']) > threshold)).astype(int)

                if 'fock_saiao' in key_list: 
                    A = (np.abs(mlf['fock_saiao']) > threshold).astype(int)

                data = build_pyg_data_from_mats(A, M_list, len(key_list), remove_self=remove_self, undirected=undirected)
                data.C_ao_saiao = mlf['C_ao_saiao']
                data.fock_saiao = mlf['fock_saiao']

                if "fock_hyb_saiao" in key_list:
                    data.fock_hyb_saiao = mlf['fock_hyb_saiao']

                print(i,j,pt_file)
                yield data

        except Exception as e:
            print(f"[SKIP] {pt_file}: {e}", flush=True)
            continue

def save_graphs_sharded(data_iter, out_dir, shard_size=1000, prefix="shard"):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []  # 每个元素: (shard_id, offset_in_shard)
    shard = []
    shard_ao_saiao = []
    shard_id = 0
    total = 0

    def flush(shard, shard_id):
        fpath = out_dir / f"{prefix}_{shard_id:06d}.pt"
        torch.save(shard, fpath)
        return fpath

    for data in data_iter:
        shard.append(data)
        index.append((shard_id, len(shard) - 1))
        total += 1

        if len(shard) >= shard_size:
            flush(shard, shard_id)
            shard_id += 1
            shard = []

            if shard_id % 10 == 0:
                torch.save(index, out_dir / "index.pt")
                print(f"Saved {total} graphs, shards={shard_id}")

    if shard:
        flush(shard,  shard_id)
        shard_id += 1

    torch.save(index, out_dir / "index.pt")
    print(f"Done. Total graphs={total}, total shards={shard_id}, out={out_dir}")

if __name__ == "__main__":
    pt_files = batch_folder(input_dir="./output_data",glob_pattern="data*.pt")
    key_list= ['dm_saiao','fock_saiao','vj_saiao','vxc_saiao','Sao_saiao','fock_hyb_saiao']
    gen = mlf2graph.data_generator_from_pt(pt_files, build_pyg_data_from_mats, key_list, threshold = 1e-7, remove_self = True, undirected = True, verbose = False)
    save_graphs_sharded(gen, "./output_data", shard_size=1000, prefix="shard")


