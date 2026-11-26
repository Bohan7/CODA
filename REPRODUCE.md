# Main experiments: settings and commands

All commands are run from the repository root after the steps in the README (environment, GOOD data,
checkpoints). The document covers the CODA experiments of the main table (Table 1).

## 1. Settings

### 1.1 Generation (stage 1)

One score network and one time-dependent classifier are released per split. All samplers use 1000 reverse
steps and classifier guidance on the node features only (`weight_adj_classifier = 0`).

| split | config | score net (layers / epochs) | classifier (layers) | SDE X / A | solver | r₁ | generated set |
|---|---|---|---|---|---|---|---|
| Motif-basis | `motif_basis.yaml` | 8 / 28,000 | 5 | VP(0.1, 1) / VP(0.1, 1) | Euler–Maruyama + Langevin (snr 0.2, scale 0.7) | 0.5 | 18,000 graphs, 3 classes |
| Motif-size | `motif_size.yaml` | 8 / 28,000 | 3 | VP(0.1, 1) / VP(0.1, 1) | Euler–Maruyama + Langevin (snr 0.2, scale 0.7) | 0.5 | 18,000 graphs, 3 classes |
| CMNIST-color | `cmnist_color.yaml` | 8 / 20,000 | 3 | VP(0.1, 1) / VP(0.1, 1) | Euler–Maruyama + Langevin (snr 0.2, scale 0.7) | 0.5 | 42,000 graphs, 10 classes |
| SST2-length | `sst2_length.yaml` | 8 / 2,000 | 3 | VP(0.1, 1) / VP(0.1, 1) | Euler–Maruyama | 0.5 | 25,000 graphs, 2 classes |
| HIV-scaffold | `hiv_scaffold.yaml` | 9 / 1,000 | 1 | VP(0.1, 1) / VE(0.2, 1) | reverse diffusion | 0.1 | 25,000 molecules, inactive : active = 24 : 1 |
| HIV-size | `hiv_size.yaml` | 9 / 500 | 1 | VP(0.1, 1) / VE(0.2, 1) | reverse diffusion | 0.1 | 26,000 molecules, inactive : active = 25 : 1 |

**Exploration.** Two knobs control how far the generated set departs from the training distribution:

* `sampler.ood` is the exploration strength λ of the paper: both the score and the classifier gradient are
  scaled by `1 - sqrt(λ)`. It is used on Motif-basis, CMNIST-color and HIV-scaffold.
* `sampler.ood_size` shifts the largest node count and replaces the empirical node-count distribution by
  `Uniform{data.min_node_num, …, max_node_num + ood_size}`. On the size / length splits exploration is
  realised through this node-count prior with `sampler.ood = 0`.

| split | explored values | reported setting (config default) |
|---|---|---|
| Motif-basis | `sampler.ood` ∈ {0.01, …, 0.05} | 0.04 |
| CMNIST-color | `sampler.ood` ∈ {0.05, 0.1} | 0.05 |
| HIV-scaffold | `sampler.ood` ∈ {0.1, 0.2, 0.3}, node-count prior Uniform{2, …, 30}, i.e. at most 30 atoms (`sampler.ood_size=-20`) | 0.1 |
| Motif-size | `sampler.ood_size` ∈ {1, …, 5} (the paper reports it as λ = 0.01 per added node) | 2 |
| SST2-length | `sampler.ood_size` ∈ {1, …, 14} (λ = 0.01 per added node) | 6 |
| HIV-size | `sampler.ood_size` ∈ {-10, …, -50} (the paper reports it as λ = 0.01 per 10 removed nodes): node-count prior Uniform{`data.min_node_num`, …, 70 + ood_size}, i.e. at most 60, 50, 40, 30 or 20 atoms (training molecules have up to 70) | -30 with `data.min_node_num` = 20: Uniform{20, …, 40}, i.e. at most 40 atoms |

`sampler.ood=0 sampler.ood_size=0` turns exploration off, which gives plain class-conditional sampling with
classifier guidance (GDSS-CC in the paper).

**GPU memory.** Sampling memory grows with `batch_size × max_node_num²`. The default batch sizes were chosen
for a 141 GB GPU (NVIDIA H200); Motif-basis, whose classifier has 5 layers, occupies about 130 GB of it at the
default batch of 3,000. Reduce `sample.batch_size` on smaller GPUs (`sample.n_graphs` must be a multiple of
it). The batch size changes which random numbers a graph receives, not the distribution of the generated set.

### 1.2 Downstream classification (stage 2)

Defaults per split (`classification/data.py`) are the settings of the main results (Table 1); 10 seeds: seed,
seed + 10, …

| split | backbone | layers | epochs | lr | lr × 0.1 at epoch | batch | variance penalty | optimiser | first seed |
|---|---|---|---|---|---|---|---|---|---|
| Motif-basis | GIN | 5 | 100 (selection after epoch 80) | 1e-3 | 80 | 32 | 0.5 | AdamW (amsgrad) | 676 |
| Motif-size | GIN | 5 | 100 | 1e-3 | – | 32 | 1.0 | AdamW (amsgrad) | 676 |
| CMNIST-color | GIN-Virtual | 5 | 100 | 1e-3 | – | 128 | 0 | AdamW (amsgrad) | 676 |
| SST2-length | GIN-Virtual | 3 | 200 | 1e-3 | 150 | 32 | 0 | AdamW (amsgrad) | 676 |
| HIV-scaffold | GIN-Virtual | 3 | 200 | 1e-4 | 150 | 32 | 0 | Adam | 133 |
| HIV-size | GIN-Virtual | 3 | 100 | 1e-3 | – | 32 | 0 | Adam | 133 |

Hidden size 300, dropout 0.5, no weight decay, mean pooling. On SST2-length the depth, epochs, learning-rate
schedule and batch size are those of the ERM configuration of the GOOD benchmark.
The variance penalty (`--var_weight`) is the variance between the mean loss on the original and on the generated
graphs of a batch, added to the loss.

Use of the generated set:

| split | generated graphs that enter training |
|---|---|
| Motif-basis, Motif-size, CMNIST-color, SST2-length | graphs with confidence ≥ 0.9 |
| HIV-scaffold | confidence > 0.9, then the 12,000 / 500 most confident inactive / active molecules |
| HIV-size | the 5,000 / 200 most confident inactive / active molecules (no confidence threshold) |

Generated graphs without edges are skipped; generated graphs with several connected components are joined by
one random edge between consecutive components. The confidence is p(y | G) under the time-dependent classifier
at the end of sampling (`<set>_confidences.pkl`).

Every default can be changed on the command line: `--backbone`, `--model_layer`, `--epochs`, `--milestones`,
`--test_epoch`, `--lr`, `--weight_decay`, `--dropout_rate`, `--batch_size`, `--var_weight`.

## 2. Commands

```bash
S=generation/samples
R=results
```

### Table 1 (CODA, checkpoint selected on OOD validation)

```bash
# stage 1: every config holds the exploration setting reported in the paper
(cd generation && python main.py sample --config configs/motif_basis.yaml  --out samples/motif_basis/lam0.04)
(cd generation && python main.py sample --config configs/motif_size.yaml   --out samples/motif_size/size2)
(cd generation && python main.py sample --config configs/cmnist_color.yaml --out samples/cmnist_color/lam0.05)
(cd generation && python main.py sample --config configs/sst2_length.yaml  --out samples/sst2_length/size6)
(cd generation && python main.py sample --config configs/hiv_scaffold.yaml --out samples/hiv_scaffold/lam0.1_n2-30)
(cd generation && python main.py sample --config configs/hiv_size.yaml     --out samples/hiv_size/n20-40)
# other points of the grids, e.g.
(cd generation && python main.py sample --config configs/motif_basis.yaml  --out samples/motif_basis/lam0.02 --set sampler.ood=0.02)
(cd generation && python main.py sample --config configs/sst2_length.yaml  --out samples/sst2_length/size9  --set sampler.ood_size=9)

# stage 2: 10 seeds
(cd classification && python train.py --dataset motif_basis  --aug ../$S/motif_basis/lam0.04.pkl        --trials 10 --out ../$R/motif_basis/coda/results.json)
(cd classification && python train.py --dataset motif_size   --aug ../$S/motif_size/size2.pkl           --trials 10 --out ../$R/motif_size/coda/results.json)
(cd classification && python train.py --dataset cmnist_color --aug ../$S/cmnist_color/lam0.05.pkl       --trials 10 --out ../$R/cmnist_color/coda/results.json)
(cd classification && python train.py --dataset sst2_length  --aug ../$S/sst2_length/size6.pkl          --trials 10 --out ../$R/sst2_length/coda/results.json)
(cd classification && python train.py --dataset hiv_scaffold --aug ../$S/hiv_scaffold/lam0.1_n2-30.txt  --trials 10 --out ../$R/hiv_scaffold/coda/results.json)
(cd classification && python train.py --dataset hiv_size     --aug ../$S/hiv_size/n20-40.txt            --trials 10 --out ../$R/hiv_size/coda/results.json)

# the same pipeline without generated graphs (ERM)
(cd classification && python train.py --dataset motif_basis --trials 10 --out ../$R/motif_basis/erm/results.json)

python scripts/collect_results.py $R                                     # mean ± std over seeds
python scripts/significance_test.py $R/motif_basis/coda/results.json $R/motif_basis/erm/results.json
```

The baselines of the paper are not part of this repository; their sources are listed in the appendix of the
paper (Baseline settings).

### Distribution distance, label preservation, molecule metrics

```bash
cd generation
# MMD-RBF on random-GIN embeddings between the training graphs and generated sets
python evaluation/mmd_rbf.py --ref data/good_motif-basis-covariate.pkl:train --gen samples/motif_basis/lam*.pkl --min_conf 0.9
# E[p(y | G)] under the time-dependent classifier
python evaluation/label_preservation.py samples/motif_basis/lam*.pkl
# molecules: validity, FCD to the training molecules, E[p(y | G)]
python evaluation/mol_metrics.py --train_smiles data/goodhiv_train_smiles.txt samples/hiv_scaffold/*.txt
```

* Ablation of the two guidance terms: `--set sampler.ood=0` removes the exploration term,
  `--set sampler.weight_x_classifier=0` removes the classifier guidance.
* In-distribution test results: `train.py --test_split id_test --select id_val`.
* Generate-then-filter (GDSS-CCD / GDSS-CCM in the paper): `train.py --ood_filter density|mahalanobis` ranks
  the generated graphs by an OOD score on graph embeddings and keeps the top half (`--ood_filter_keep_ratio`).
  The embeddings come from the encoder passed as `--pretrain_ckpt` (a model saved with `--save_ckpt`), or from
  a randomly initialised encoder when none is given. `--ood_filter_per_class` applies the threshold within
  every class.

## 3. Differences from the code used for the paper

The release is a cleaned version of the research code: the samplers of all datasets and the six downstream
scripts were merged, unused branches were removed, and checkpoints were re-saved without pickled Python
objects. With the same seed the release reproduces the research code bit for bit: identical generated
graphs / molecules on every dataset (on CPU and on GPU), and identical GNN weights after two epochs of
downstream training on every split (on CPU; GIN-Virtual on HIV and SST2 was compared with a single thread,
because multi-threaded CPU training of these models is not reproducible from run to run). The following
behaviour differs on purpose:

* Generated molecules whose SMILES is empty are dropped together with their label and confidence (the
  research code dropped only the SMILES, which shifted the alignment of the remaining entries).
* The `label` attribute stored inside the generated networkx graphs of CMNIST / SST2 is the graph's own
  label (it was a copy of one fixed label; training always used `<set>_labels.pkl`, which is unchanged).
* Binary labels of generated SST2 graphs are stored as floats, which avoids a type error when a batch
  contains generated graphs only. A final training batch that would hold a single graph is dropped.
* The molecule sampler no longer reads the training SMILES before sampling. This changes the state of the
  random generator, so a fixed seed gives a different (equally distributed) HIV set than the research code.
* Default sampling batch sizes fit a single GPU; the paper's sets were sampled with larger batches.
* The generate-then-filter baselines score every generated graph and threshold over the whole set by
  default. The research scripts did not score the last incomplete batch of 64 generated graphs on CMNIST,
  SST2 and HIV, and thresholded within every class on Motif-basis (`--ood_filter_per_class`).
* Data preparation with RDKit 2025.09 kekulises 3 of the 22,889 HIV-scaffold training molecules (6 of
  24,926 for HIV-size) differently from the files the released checkpoints were trained on.
