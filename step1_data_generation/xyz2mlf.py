import saao_simple as saao
import numpy as np
import torch
import os
from pathlib import Path
import random

def is_invalid(data, key_list):
    if not isinstance(data, dict):
        return True, f"output is not dict: {type(data).__name__}"

    for key in key_list:
        if key not in data:
            return True, f"missing key: {key}"

        val = data[key]
        if val is None:
            return True, f"{key} is None"

        if isinstance(val, str):
            if val.strip() == "":
                return True, f"{key} is empty string"

        elif isinstance(val, np.ndarray):
            if val.size == 0:
                return True, f"{key} is empty ndarray"
            if np.isnan(val).any():
                return True, f"{key} contains NaN"
            if np.isinf(val).any():
                return True, f"{key} contains Inf"

        elif isinstance(val, (list, tuple)):
            if len(val) == 0:
                return True, f"{key} is empty list/tuple"
            try:
                arr = np.asarray(val, dtype=float)
                if np.isnan(arr).any():
                    return True, f"{key} contains NaN"
                if np.isinf(arr).any():
                    return True, f"{key} contains Inf"
            except Exception:
                pass

        else:
            try:
                if np.isnan(val):
                    return True, f"{key} is NaN"
            except Exception:
                pass
            try:
                if np.isinf(val):
                    return True, f"{key} is Inf"
            except Exception:
                pass
    return False, "OK"


def batch_folder(input_dir, glob_pattern="*.xyz"):
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")

    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        print(f"No files matched {glob_pattern} in {input_dir}")
        return

    return [str(f.resolve()) for f in files]

def save_mlfs(gen, xyz_files, key_list, save_dir = "./output_data", log_txt = "process_log.txt", bad_txt = "bad_files.txt" , batch_size = 100):
    os.makedirs(save_dir, exist_ok=True)
    batch_data = []
    batch_id = 0
    bad_files = []

    with open(log_txt, "w") as logf:
        for idx, data in enumerate(gen):
            xyz_file = xyz_files[idx]
            invalid, reason = is_invalid(data, key_list)
 
            if invalid:
                msg = f"[INVALID] {xyz_file} | {reason}"
                print(msg)
                logf.write(msg + "\n")
                bad_files.append(xyz_file)
                continue
 
            msg = f"[OK] {xyz_file}"
            print(msg)
            logf.write(msg + "\n")
 
            batch_data.append(data)
 
            if len(batch_data) >= batch_size:
                save_path = os.path.join(save_dir, f"data_batch_{batch_id:04d}.pt")
                torch.save(batch_data, save_path)
 
                msg = f"[SAVE] {save_path} | {len(batch_data)} samples"
                print(msg)
                logf.write(msg + "\n")
 
                batch_data = []
                batch_id += 1

    if batch_data:
        save_path = os.path.join(save_dir, f"data_batch_{batch_id:04d}.pt")
        torch.save(batch_data, save_path)

        with open(log_txt, "a") as logf:
            msg = f"[SAVE] {save_path} | {len(batch_data)} samples"
            print(msg)
            logf.write(msg + "\n")

    with open(bad_txt, "w") as f:
        for fn in bad_files:
            f.write(fn + "\n")

if __name__ == "__main__":
    xyz_files = batch_folder(input_dir="../qm9_xyz_converted",glob_pattern="*.xyz")
    random.seed(42) 
    random.shuffle(xyz_files)
    key_list= ['dm_saiao','fock_saiao','vj_saiao','vxc_saiao','Sao_saiao','fock_hyb_saiao']
    gen = saao.construct_mlf_saiao(xyz_files[:2], basis = "def2-SVP", init = "minao", 
                        gga_xc = "PBE", gga_conv_tol = 1e-6, gga_max_cycle = 500, 
                        get_hyb = True, hyb_xc = "PBE0", hyb_conv_tol = 1e-9, hyb_max_cycle = 500,
                        threshold = 1e-6)
    save_mlfs(gen, xyz_files, key_list, save_dir = "./output_data", log_txt = "process_log.txt", bad_txt = "bad_files.txt" , batch_size = 100)
