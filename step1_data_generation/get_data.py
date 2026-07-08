import random
from saao_simple import construct_mlf_saiao
from xyz2mlf import save_mlfs
from mlf2graph import batch_folder, data_generator_from_pt, build_pyg_data_from_mats, save_graphs_sharded

#Step 1: Get mlfs
#key_list= ['dm_saiao','fock_saiao','vj_saiao','vxc_saiao','Sao_saiao','fock_hyb_saiao']

#xyz_files = batch_folder(input_dir="./qm9_xyz_subset",glob_pattern="*.xyz")
#random.seed(42) 
#random.shuffle(xyz_files)
#gen = construct_mlf_saiao(xyz_files[:200], basis = "def2-SVP", init = "minao", 
#                    gga_xc = "PBE", gga_conv_tol = 1e-6, gga_max_cycle = 500, 
#                    get_hyb = True, hyb_xc = "PBE0", hyb_conv_tol = 1e-6, hyb_max_cycle = 500,
#                    verbose = True)
#save_mlfs(gen, xyz_files, key_list, save_dir = "./output_data", log_txt = "process_log.txt", bad_txt = "bad_files.txt", batch_size = 100)

#Step 2: Get Graphs
key_list= ['dm_saiao','fock_saiao','vj_saiao','vxc_saiao','Sao_saiao','fock_hyb_saiao']
pt_files = batch_folder(input_dir="./output_data",glob_pattern="data*.pt")
gen = data_generator_from_pt(pt_files, build_pyg_data_from_mats, key_list, threshold = 1e-5, remove_self = True, undirected = True, verbose = False)
save_graphs_sharded(gen, "./output_data", shard_size=1000, prefix="shard")

