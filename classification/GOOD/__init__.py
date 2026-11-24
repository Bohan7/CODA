r"""Minimal subset of the GOOD benchmark (https://github.com/divelab/GOOD, GPL-3.0) used by CODA:
the four dataset loaders (GOOD-Motif, GOOD-CMNIST, GOOD-HIV, GOOD-SST2) and the GIN / GIN-Virtual / GCN / GAT
backbones. Adapted for CODA (GAT backbone added, GCN extended to molecules, config system reduced to type
aliases); see `LICENSE` in this directory and `THIRD_PARTY_NOTICES.md` in the repository root.
"""
from .utils.register import register
