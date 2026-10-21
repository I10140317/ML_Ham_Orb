# Third-Party Code Attribution

## MLGF: Machine Learning Many-Body Green's Functions

Parts of the SAIAO construction and electronic-structure preprocessing routines in this repository are derived from the MLGF project developed by Zhu and co-workers.

**Original repository:** https://github.com/ZhuGroup-Yale/mlgf

**Original project:** Machine Learning Many-Body Green's Functions and Self-Energy

**Original contributors:** Christian Venturella, Jiachen Li, Christopher Hillenbrand, Tianyu Zhu, and other contributors as identified in the upstream repository.

**License:** GNU General Public License version 3 (GPLv3), subject to the terms specified in the original source files.

### Reused Components

The following components are based on or adapted from the MLGF implementation:

- Construction of intrinsic atomic orbital and projected atomic orbital (IAO+PAO) representations.
- Symmetry adaptation for constructing the SAIAO basis.
- Related electronic-structure preprocessing and matrix transformation routines.

The specific reused files, their upstream counterparts, and any modifications are documented in the source files and repository history.

### Original Contributions

The graph neural network implementation for predicting hybrid-functional Kohn–Sham Hamiltonians was developed by the authors of ML_Orb.

The implementation employs edge-dependent gated message passing with fixed edge embeddings during graph convolution. While it builds upon established SAIAO representations and orbital-graph learning concepts, its GNN architecture and implementation differ from those used in MLGF.

### Licensing and Attribution

The original copyright and license notices for reused MLGF code are retained.

The corresponding source code and modifications are made available in this repository under the applicable GPLv3 licensing terms.

We gratefully acknowledge the contributions of Zhu and co-workers to the development of the SAIAO representation and the MLGF software.
