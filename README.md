# CODA: Score-based Conditional Out-of-Distribution Augmentation for Graph Covariate Shift （KDD 2027）

Official implementation of **CODA**, accepted at **KDD 2027**:

> **Score-based Conditional Out-of-Distribution Augmentation for Graph Covariate Shift**
> Bohan Wang, Yurui Chang, Wei Jin, Lu Lin
> ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2027 · [arXiv:2410.17506](https://arxiv.org/abs/2410.17506)

CODA mitigates graph covariate shift by *generating* additional training graphs. A score-based graph diffusion
model is steered by a time-dependent classifier, so that generated graphs keep the label-determining
(stable) patterns, and by an exploration strength λ that tempers the class-conditional score and moves
samples toward low-density regions of the training distribution:

```
score_CODA(G_t) = (1 - sqrt(λ)) * [ s_θ(G_t) + α_t ∇ log p_t(y | G_t) ]
```

The generated graphs are filtered by classifier confidence and added to the training set of a standard GNN.

```
           stage 1: generation/                                  stage 2: classification/
 training graphs ──► score network s_θ ─┐
                                        ├─► guided reverse SDE ─► augmentation set ─┐
 labelled graphs ──► classifier φ_t  ───┘      (λ, class y)                         ├─► GNN (GIN / vGIN / GCN / GAT)
 original training graphs ──────────────────────────────────────────────────────────┘
```

## Repository layout

```
generation/              stage 1: score network, time-dependent classifier, CODA sampler
  main.py                  entry point: train_score | train_classifier | sample
  configs/                 one config per dataset split
  sde.py  solver.py        SDEs and the predictor-corrector solver with CODA guidance
  sampler.py  trainer.py   sampling and training loops
  models/  utils/          graph transformer, data and molecule utilities
  data/                    scripts that convert GOOD datasets into the generator's training files
  evaluation/              MMD-RBF, FCD / validity, label preservation
classification/          stage 2: downstream OOD graph classification
  train.py                 trains a GNN on original + generated graphs (also ERM and GDSS-CCD/CCM baselines)
  data.py  ood_filter.py   dataset settings, augmentation loading, post-hoc OOD filters
  GOOD/                    minimal subset of the GOOD benchmark (datasets and GNN backbones)
scripts/                 checkpoint download, significance test, result collection
tests/                   unit tests (no data required)
REPRODUCE.md             settings and commands of the main experiments
```

## Installation

```bash
conda env create -f environment.yml
conda activate coda
pip install --no-deps -r requirements-extra.txt     # FCD metric, GOOD-SST2 loader
```

The code was run with Python 3.9, PyTorch 2.1.0 (CUDA 12.1), PyTorch Geometric 2.2.0 and RDKit 2025.09.
PyTorch Geometric must be 2.2.x: the GOOD backbones in `classification/GOOD` rely on its message-passing API.

## Data

All experiments use the covariate-shift splits of the [GOOD benchmark](https://github.com/divelab/GOOD):
GOOD-Motif (basis, size), GOOD-CMNIST (color), GOOD-HIV (scaffold, size) and GOOD-SST2 (length).
GOOD downloads each dataset into `data/GOOD/` the first time it is loaded.

The generator is trained on dense tensors derived from the GOOD training split. These files are needed to
(re)train the score network or the classifier and by the scripts in `generation/evaluation`; sampling from the
released checkpoints and the downstream stage do not need them.

```bash
cd generation
python data/prepare_good.py --dataset motif  --domain basis      # -> data/good_motif-basis-covariate.pkl
python data/prepare_good.py --dataset motif  --domain size
python data/prepare_good.py --dataset cmnist --domain color
python data/prepare_good.py --dataset sst2   --domain length
python data/prepare_good_hiv.py --domain scaffold                # -> data/goodhiv_*_kekulized.npz, data/goodhiv_train_smiles.txt
python data/prepare_good_hiv.py --domain size                    # -> data/goodhivsize_*_kekulized.npz
cd ..
```

## Pretrained checkpoints

The checkpoints of all six splits are hosted on the Hugging Face Hub:
[BobbbWang/CODA](https://huggingface.co/BobbbWang/CODA).

```bash
# from the repository root
python scripts/download_checkpoints.py            # -> generation/checkpoints/<split>/{score,classifier}.pth
```

Each dataset directory holds the score network (`score.pth`), the time-dependent classifier
(`classifier.pth`) and the node-count statistics of the training graphs (`train_node_flags.pt`), so sampling
does not need the generator's training files.

## Quick start

```bash
# stage 1: generate 25,000 molecules for GOOD-HIV (scaffold); the config holds the reported setting, λ = 0.1
cd generation
python main.py sample --config configs/hiv_scaffold.yaml --out samples/hiv_scaffold/lam0.1_n2-30

# stage 2: train GIN-Virtual on the original + generated molecules, 10 seeds
cd ../classification
python train.py --dataset hiv_scaffold --aug ../generation/samples/hiv_scaffold/lam0.1_n2-30.txt \
                --trials 10 --out ../results/hiv_scaffold/coda/results.json
```

`train.py` prints, for every seed, the OOD test score at the best epoch of the OOD-validation split (the
GOOD protocol) and at the best epoch of the ID-validation split.

### Generation

| setting (`--set key=value`) | meaning |
|---|---|
| `sampler.ood` | exploration strength λ ∈ [0, 1); λ = 0 is classifier-guided class-conditional generation (the GDSS-CC baseline) |
| `sampler.ood_size`, `data.min_node_num` | shift of the largest node count and smallest node count; when `ood_size != 0` node counts are drawn uniformly from `{min_node_num, …, max_node_num + ood_size}` instead of from the training graphs |
| `sampler.weight_x_classifier`, `sampler.weight_adj_classifier` | classifier-guidance weights r₁ (node features) and r₂ (adjacency) |
| `sample.n_graphs`, `sample.batch_size`, `sample.seed` | size of the generated set, sampling batch, random seed |

Every config defaults to the exploration setting reported in the paper; `--set` changes single entries (an unknown key
is an error). A generated set consists of `<out>.pkl` (networkx graphs) or `<out>.txt` (SMILES),
`<out>_labels.pkl` and `<out>_confidences.pkl`. Sampling writes a per-round checkpoint
(`<out>.partial.pkl`) and resumes from it when it is restarted with the same settings.

Training from scratch (needs the files written by `data/prepare_good*.py`):

```bash
python main.py train_score      --config configs/motif_basis.yaml --out_dir runs/motif_basis   # -> runs/motif_basis/score.pth
python main.py train_classifier --config configs/motif_basis.yaml --out_dir runs/motif_basis   # -> runs/motif_basis/classifier.pth
python main.py sample --config configs/motif_basis.yaml --out samples/motif_basis/own \
       --set ckpt.score=runs/motif_basis/score.pth ckpt.classifier=runs/motif_basis/classifier.pth
```

### Downstream classification

| option | meaning |
|---|---|
| `--dataset` | `motif_basis`, `motif_size`, `cmnist_color`, `sst2_length`, `hiv_scaffold`, `hiv_size` |
| `--aug` | generated set; omit for ERM |
| `--select` | checkpoint selection split: `val` (OOD validation, default) or `id_val` |
| `--backbone` | `GIN`, `vGIN` (GIN with virtual node), `GCN`, `GAT`; default per dataset |
| `--ood_filter` | `density` (GDSS-CCD) or `mahalanobis` (GDSS-CCM) post-hoc filter of a λ = 0 set |
| `--ood_filter_per_class` | apply the filter threshold within every class instead of over the whole set |
| `--model_layer`, `--epochs`, `--milestones`, `--test_epoch`, `--lr`, `--batch_size`, `--weight_decay`, `--dropout_rate` | GNN depth and optimisation settings |
| `--var_weight` | weight of the variance of the mean losses on original and on generated graphs (V-REx penalty; 0 = off; default per dataset) |
| `--seed`, `--trials` | first seed and number of seeds (seed, seed + 10, …) |

The per-dataset defaults (backbone, depth, epochs, learning rate, batch size, variance penalty, confidence filter) are listed in
`classification/data.py`.

## Experiments of the paper

See [REPRODUCE.md](REPRODUCE.md) for the settings and commands of the main experiments (Table 1).

## Citation

```bibtex
@inproceedings{wang2027score,
  title     = {Score-based Conditional Out-of-Distribution Augmentation for Graph Covariate Shift},
  author    = {Wang, Bohan and Chang, Yurui and Jin, Wei and Lin, Lu},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2027}
}
```

## Acknowledgements

This code builds on [GDSS](https://github.com/harryjo97/GDSS), [DiGress](https://github.com/cvignac/DiGress),
[MoFlow](https://github.com/calvin-zcx/moflow) and the [GOOD benchmark](https://github.com/divelab/GOOD).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This repository is released under the GNU General Public License v3.0 ([LICENSE](LICENSE)). Third-party code and
its licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
