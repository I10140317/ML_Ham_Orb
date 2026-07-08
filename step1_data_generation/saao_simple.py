import re
import os
import numpy as np
import warnings
from pyscf import gto, dft
from libdmet.basis_transform import make_basis
from pyscf.lo.iao import reference_mol
import time

def get_saao(mol, dm, force_eigv_direction = False):
    '''
    Symmetry-adapted atomic orbitals
    Ref: JCP 153, 124111 (2020)

    Return:
        C_ao_saao : (nao, nsaao) 2D array, each column corresponds to a SAAO
    '''

    # count AOs on each nl (n: 0,1,2,...; l: s, p, d...)
    nao_nl = []
    for i in range(mol.nbas):
        for k in range(mol.bas_nctr(i)):
            nao_nl.append(2 * mol.bas_angular(i) + 1)

    # AO index on each nl
    idx_tmp_start = 0
    idx_nl = []
    for i in range(len(nao_nl)):
        idx_tmp_end = idx_tmp_start + nao_nl[i]
        idx_nl.append([*range(idx_tmp_start, idx_tmp_end)])
        idx_tmp_start = idx_tmp_end

    # diagonalize each nl block of mean-field density matrix
    C_ao_saao = np.zeros_like(dm)
    for i in range(len(nao_nl)):
        ix = np.ix_(idx_nl[i], idx_nl[i])
        e_nl, y_nl = np.linalg.eigh(dm[ix])
        if force_eigv_direction:
            slice_bool = np.sum(y_nl, axis = 0) < 0
            y_nl[:,slice_bool] = -y_nl[:,slice_bool]
        C_ao_saao[ix] = y_nl

    return C_ao_saao

def get_C_ao_iao(mf, minao, minao_val=None, minao_core=None):
    '''
    Intrinsic atomic orbital + projected atomic orbital (IAO+PAO) basis
    Ref: JCTC 9, 4834-4843 (2013)

    Return:
        C_ao_iao : (nao, niao) 2D array, each column corresponds to a IAO or PAO
                    Note: the orbital order is the same as original AO basis
    '''

    if minao_val is None and minao_core is None:
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, tol = 1e-6)
        mol = mf.mol
        pmol = reference_mol(mol, minao=minao)
        B1_labels = mol.ao_labels()
        B2_labels = pmol.ao_labels()
        iao_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels)]
        iao_virt_idx = iao_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(iao_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]
    else:
        assert (minao_val is not None)
        assert (minao_core is not None)
        from libdmet.lo import iao
        mol = mf.mol

        print('Building pmol_core and pmol_val', flush = True)
        pmol_core, pmol_val = iao.build_pmol_core_val(mol, minao_core, minao_val)
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, pmol_val=pmol_val,
                                                  pmol_core=pmol_core, tol = 1e-6)
        B1_labels = mol.ao_labels()
        B2_labels = pmol_core.ao_labels()
        B3_labels = pmol_val.ao_labels()
        core_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        val_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B3_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels and label not in B3_labels)]
        core_val_virt_idx = core_idx + val_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(core_val_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]

    return C_ao_iao_ordered

def get_chk_saiao(mf, fock_ao, minao = 'minao', force_eigv_direction = False, val_core_dats = None):
    """wrapper for localized IAO transform and symmetrization rotations 

    Args:
        mf (pyscf.mf): mean field object like DFT
        fock_ao (float64): ao basis fock
        minao (str, optional): which definition of MINAO to use. Defaults to 'minao'.
        force_eigv_direction (bool, optional): force the eigenvalue direction. Defaults to False.
        val_core_dats (list, optional): data for redefining MINAO native to pyscf. Defaults to None.

    Returns:
        tuple: C_ao_iao, C_iao_saiao, fock_iao
    """    

    if val_core_dats is None:
        C_ao_iao = get_C_ao_iao(mf, minao=minao)
    else:
        C_ao_iao = get_C_ao_iao(mf, minao=minao, minao_val=val_core_dats[0], minao_core=val_core_dats[1])

    fock_iao = C_ao_iao.T @ fock_ao @ C_ao_iao
    C_iao_saiao = get_saao(mf.mol, fock_iao, force_eigv_direction = force_eigv_direction)
    
    assert(np.linalg.norm(np.dot(C_iao_saiao.T, C_iao_saiao)-np.eye(fock_iao.shape[0]))<1e-8)
    return C_ao_iao, C_iao_saiao, fock_iao

def fast_ao_fock(mo_energy, mo_coeff, ovlp):
    C_ao_mo_f = ovlp @ mo_coeff
    return C_ao_mo_f @ np.diag(mo_energy) @ C_ao_mo_f.T # Fock matrix in AO basis from diagonal MO basis

def get_saiao_features(mlf, get_veff = False, get_vj = True, get_vxc = True):
    basis_name = 'saiao'

    if 'fock' in mlf:
        C_ao_saiao = mlf['C_ao_saiao']
        mo_energy = mlf['mo_energy']
        mo_coeff = mlf['mo_coeff']
        S_ao = mlf['ovlp']
        dm = mlf['dm']
                
        # feature 1: density matrix
        dm_saiao = np.linalg.multi_dot((C_ao_saiao.T, S_ao, dm, S_ao, C_ao_saiao))
        nocc = mlf['nocc']
        nelectron = nocc*2
        abs_diff_particle_number = abs(np.trace(dm_saiao)-nelectron)
        if abs_diff_particle_number > 1e-5:
            warnings.warn(f'dm_saiao particle number diff {abs_diff_particle_number:0.6e}')
 
        # feature 2 : Fock matrix
        fock = mlf['fock']
        fock_saiao = np.linalg.multi_dot((C_ao_saiao.T, fock, C_ao_saiao))
 
        # feature 3 : Vxc
        if get_vxc:
            vxc = mlf['vxc']
            vxc_saiao = np.linalg.multi_dot((C_ao_saiao.T, vxc, C_ao_saiao))
            mlf[f'vxc_{basis_name}'] = vxc_saiao
 
        # feature 4 : Vj
        if get_vj:
            vj = mlf['vj']
            vj_saiao = np.linalg.multi_dot((C_ao_saiao.T, vj, C_ao_saiao))
            mlf[f'vj_{basis_name}'] = vj_saiao
 
        # feature X : Veff
        if get_veff:
            veff = mlf['veff']
            veff_saiao = np.linalg.multi_dot((C_ao_saiao.T, veff, C_ao_saiao))
            mlf[f'veff_{basis_name}'] = veff_saiao

        hcore = mlf['hcore']
        hcore_saiao = np.linalg.multi_dot((C_ao_saiao.T, hcore, C_ao_saiao))
        mlf[f'hcore_{basis_name}'] = hcore_saiao

        # feature 5 : Overlap matrices
        Sao_saiao = np.linalg.multi_dot((C_ao_saiao.T, S_ao, C_ao_saiao))

        mlf[f'dm_{basis_name}'] = dm_saiao
        mlf[f'fock_{basis_name}'] = fock_saiao
        mlf[f'Sao_{basis_name}'] = Sao_saiao

    # target : pbe0 Fock matrix
    if 'fock_hyb' in mlf and 'fock' in mlf:
        fock_hyb = mlf['fock_hyb']
        fock_hyb_saiao = np.linalg.multi_dot((C_ao_saiao.T, fock_hyb, C_ao_saiao))
        mlf[f'fock_hyb_{basis_name}'] = fock_hyb_saiao
    
    return mlf

def construct_mlf_saiao(xyz_files, basis = "def2-SVP", init = "minao", 
                        gga_xc = "PBE", gga_conv_tol = 1e-6, gga_max_cycle = 500, 
                        get_veff = True, get_vj = True, get_vxc = True,
                        get_hyb = True, hyb_xc = "PBE0", hyb_conv_tol = 1e-9, hyb_max_cycle = 500,
                        verbose = True):

    for i, xyz_file in enumerate(xyz_files):
        if verbose:
            print(f"[{i}] {xyz_file}", flush=True)
        try:
            mol = gto.M(
                atom=xyz_file,
                basis=basis,
            )
            if verbose:
                print('start',time.time())

            mf = dft.RKS(mol)
            mf.xc = gga_xc
            mf.init_guess = init
            mf.max_cycle = gga_max_cycle
            mf.conv_tol = gga_conv_tol
            mf = mf.density_fit()
            mf.kernel()
  
            if verbose:
                print("pbe done",time.time())
            mlf = {}
  
            # DFT/HF calculation outputs, cheap stuff below
            mlf['e_mf'] = mf.e_tot                         # DFT energy    
            nocc =  mol.nelectron // 2           
            mlf['nocc'] = nocc                             # occupation number/number of electrons
            mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
            mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
            mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
            mlf['dm'] = np.asarray(mf.make_rdm1())         # mean field density matrix
            mlf['fock'] = np.asarray(fast_ao_fock(mlf['mo_energy'], mlf['mo_coeff'], mlf['ovlp']))

            mlf['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix

            if get_veff and not get_vxc:
                mlf['veff'] = mlf['fock'] - mlf['hcore']   # exchange-correlation matrix

            if get_vj and not get_vxc:
                mlf['vj'] = np.asarray(mf.get_j())

            if get_vxc:
                mlf['veff'] = mlf['fock'] - mlf['hcore']   # exchange-correlation matrix
                mlf['vj'] = np.asarray(mf.get_j())
                mlf['vxc'] = np.asarray(mlf['veff'] - mlf['vj'])    # exchange-correlation matrix
  
            fock_ao = mlf['fock']
            C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf, fock_ao, minao = 'minao', force_eigv_direction = True, val_core_dats = None)
            C_ao_saiao = C_ao_iao @ C_iao_saiao
            mlf['C_ao_saiao'] = C_ao_saiao
  
            if verbose:
                print("get C_ao_saiao done",time.time())
  
            if get_hyb:
                mf.xc = hyb_xc
                mf.max_cycle = hyb_max_cycle
                mf.conv_tol = hyb_conv_tol
                mf = mf.density_fit()
                mf.kernel()
   
                mlf['e_mf_hyb'] = mf.e_tot                         # DFT energy    
                mlf['mo_energy_hyb'] = np.asarray(mf.mo_energy)    # orbital energy
                mlf['mo_coeff_hyb'] = np.asarray(mf.mo_coeff)      # orbital coefficient
                mlf['fock_hyb'] = np.asarray(fast_ao_fock(mlf['mo_energy_hyb'], mlf['mo_coeff_hyb'], mlf['ovlp']))   
                if verbose:
                    print("pbe0 done",time.time())
  
            mlf = get_saiao_features(mlf, get_veff = get_veff, get_vj = get_vj, get_vxc = get_vxc)
            mlf['xyz_file']=xyz_file

            if verbose:
                print("get_features done",time.time())

            yield mlf

        except Exception as e:
            print(f"[SKIP] {xyz_file}: {e}", flush=True)
            continue

