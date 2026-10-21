# ML_Ham_Orb

**Efficient and Transferable Prediction of Hybrid-Functional Kohn–Sham Hamiltonians for Large Molecules Using Low-Cost Electronic-Structure Features and an Orbital-Graph Neural Network**

## Overview

ML_Ham_Orb is a machine-learning framework for predicting hybrid-functional Kohn–Sham Hamiltonians from lower-cost electronic-structure features. The project uses the established symmetry-adapted intrinsic atomic orbital (SAIAO) representation and an orbital-graph neural network developed for this Hamiltonian-prediction task.

## Requirements

- **PySCF** and all required dependencies.
- **libdmet**, developed by Zhi-Hao Cui: https://github.com/gkclab/libdmet_preview
- **scikit-learn**, for self-energy kernel ridge regression (KRR): https://scikit-learn.org
- **PyTorch Geometric** and all required dependencies: https://pytorch-geometric.readthedocs.io and https://pytorch.org

## Code attribution

The file `step1_data_generation/saao_simple.py` contains modified code derived from the MLGF project developed by Zhu and co-workers. The reused and modified routines support the following functions:

- Construction of intrinsic atomic orbital and projected atomic orbital (IAO+PAO) representations.
- Symmetry adaptation for constructing the SAIAO basis.
- Related electronic-structure preprocessing and matrix-transformation routines.

The original MLGF repository is available at: https://github.com/ZhuGroup-Yale/mlgf

The original copyright and license notices associated with the reused code must be retained. The modifications in `saao_simple.py` should be identified in that file, with the relevant modification date. Further details are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The graph neural network implementation for hybrid-functional Hamiltonian prediction was developed by the ML_Ham_Orb authors. It uses an edge-dependent gated message-passing architecture with fixed edge embeddings during graph convolution. This implementation builds upon established SAIAO and orbital-graph learning concepts but is distinct from the GNN implementation in MLGF.

## License

This repository includes modified code derived from MLGF, which is distributed under the GNU General Public License, version 3 (GPLv3). See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Before redistributing this repository, ensure that the repository-wide licensing arrangement is consistent with the GPLv3 obligations that apply to the modified MLGF-derived code and the combined work.

## References

### A. Prior Work and Related Methods

1. Knizia, G. Intrinsic Atomic Orbitals: An Unbiased Bridge between Quantum Theory and Chemical Concepts. *Journal of Chemical Theory and Computation* **2013**, *9* (11), 4834–4843. https://doi.org/10.1021/ct400687b.
2. Venturella, C.; Hillenbrand, C.; Li, J. C.; Zhu, T. Y. Machine Learning Many-Body Green's Functions for Molecular Excitation Spectra. *Journal of Chemical Theory and Computation* **2024**, *20*, 143–154. https://doi.org/10.1021/acs.jctc.3c01141.
3. Venturella, C.; Li, J.; Hillenbrand, C.; Leyva Peralta, X.; Liu, J.; Zhu, T. Unified Deep Learning Framework for Many-Body Quantum Chemistry via Green's Functions. *Nature Computational Science* **2025**, *5*, 502–513. https://doi.org/10.1038/s43588-025-00810-z.
4. Zhang, C.; Venturella, C.; Chen, E.; Zhu, T. Transferable Machine Learning of Electronic Hamiltonians with Superposition-of-Atomic-Potentials Features. *arXiv* **2026**, arXiv:2606.12326. https://doi.org/10.48550/arXiv.2606.12326.

### B. This Work

Liu, Xiang-Yang; Xiao, Dong-Yi; Xiang, Qing-Xin; Cui, Ganglong. *Efficient and Transferable Prediction of Hybrid-Functional Kohn–Sham Hamiltonians for Large Molecules Using Low-Cost Electronic-Structure Features and an Orbital-Graph Neural Network.* Manuscript under review.

