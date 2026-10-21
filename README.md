# ML_Orb

Efficient and Transferable Prediction of Hybrid-Functional Kohn–Sham Hamiltonians Using an Orbital-Graph Neural Network

## Overview

ML_Orb is a machine-learning framework for predicting hybrid-functional Kohn–Sham Hamiltonians from lower-cost electronic-structure features.

The framework uses the previously established symmetry-adapted intrinsic atomic orbital (SAIAO) representation and an orbital-graph neural network with edge-dependent gated message passing.

## Code Attribution

The SAIAO construction and parts of the electronic-structure preprocessing routines are derived from the MLGF project developed by Zhu and co-workers:

https://github.com/ZhuGroup-Yale/mlgf

The GNN implementation for the present Hamiltonian prediction task was developed by the ML_Orb authors. It uses a different message-passing architecture while building upon established orbital-graph learning concepts.

For details of the reused code and its licensing, please see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This repository is distributed under the applicable GNU General Public License version 3 terms. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## References

Venturella, C.; Hillenbrand, C.; Li, J.; Zhu, T. Machine Learning Many-Body Green's Functions for Molecular Excitation Spectra. J. Chem. Theory Comput. 2024, 20, 143–154.

Venturella, C.; Li, J.; Hillenbrand, C.; Leyva Peralta, X.; Liu, J.; Zhu, T. Unified Deep Learning Framework for Many-Body Quantum Chemistry via Green's Functions. Nat. Comput. Sci. 2025, 5, 502–513.
