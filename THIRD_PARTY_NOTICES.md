# Third-Party Code Notices

## MLGF

The file `step1_data_generation/saao_simple.py` contains code modified from the MLGF project developed by Zhu and co-workers:

- **Project:** MLGF — Machine Learning Many-Body Green's Functions
- **Repository:** https://github.com/ZhuGroup-Yale/mlgf
- **License:** GNU General Public License, version 3 (GPLv3), as specified by the upstream repository.

The modified routines in `saao_simple.py` support the following functions:

1. Construction of intrinsic atomic orbital and projected atomic orbital (IAO+PAO) representations.
2. Symmetry adaptation for constructing the SAIAO basis.
3. Related electronic-structure preprocessing and matrix-transformation routines.

The `saao_simple.py` file should retain all original copyright, authorship, license, and warranty notices from the upstream source. It should also prominently state that the file has been modified and include the date of modification, as required by the applicable GPLv3 terms.

## Original contributions in ML_Ham_Orb

The graph neural network implementation for predicting hybrid-functional Kohn–Sham Hamiltonians was developed by the ML_Orb authors. It employs edge-dependent gated message passing with fixed edge embeddings during graph convolution. This implementation uses the established SAIAO representation and orbital-graph concepts but is a distinct implementation from the GNN code in MLGF.

## License compliance

A copy of the GNU GPL version 3 is provided in the repository's `LICENSE` file. Redistribution of the modified MLGF-derived code, and of any combined work that is legally treated as a derivative work, must comply with the applicable GPLv3 requirements. The inclusion of this notice does not replace or modify the upstream license terms.
