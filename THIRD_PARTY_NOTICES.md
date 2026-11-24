# Third-party code

Parts of this repository are adapted from the following projects.

| Project | Used for | Files | Licence |
|---|---|---|---|
| [GDSS](https://github.com/harryjo97/GDSS) (Jo, Lee, Hwang; ICML 2022) | SDEs, predictor-corrector solver, score-matching loss, EMA, training / sampling loops | `generation/sde.py`, `solver.py`, `losses.py`, `trainer.py`, `sampler.py`, `utils/` | no licence file in the upstream repository |
| [Score SDE](https://github.com/yang-song/score_sde_pytorch) (Song et al.; ICLR 2021) | SDE definitions and EMA, via GDSS | `generation/sde.py`, `generation/utils/ema.py` | Apache-2.0 |
| [DiGress](https://github.com/cvignac/DiGress) (Vignac et al.; ICLR 2023) | graph transformer layers | `generation/models/transformer.py` | MIT, Copyright (c) 2012-2022 Clement Vignac, Igor Krawczuk, Antoine Siraudin |
| [MoFlow](https://github.com/calvin-zcx/moflow) (Zang, Wang; KDD 2020) | molecule featurisation and valency correction | `generation/utils/mol_utils.py`, `generation/data/prepare_good_hiv.py` | MIT, Copyright (c) 2020 Chengxi Zang |
| [GOOD](https://github.com/divelab/GOOD) (Gui et al.; NeurIPS 2022) | dataset loaders and GNN backbones | `classification/GOOD/` (adapted, see below; licence text included) | GPL-3.0 (code); the GOOD datasets are distributed under the MIT licence |

## Changes to the GOOD files

`classification/GOOD/` is a subset of GOOD (dataset loaders of GOOD-Motif / GOOD-CMNIST / GOOD-HIV / GOOD-SST2 and
the GIN, GIN-Virtual and GCN backbones). Changes made for CODA (2025–2026):

* `networks/models/GAT.py` was added (GAT backbone in the style of the GOOD GCN).
* `networks/models/GCNs.py` was extended to molecule inputs (atom / bond encoders).
* `utils/config_reader.py` was reduced to the type aliases used in the model signatures.
* the package `__init__.py` files import only the files listed above; a debugging import was removed from
  `data/good_datasets/good_hiv.py`.

## MIT licence (DiGress, MoFlow)

DiGress: Copyright (c) 2012-2022 Clement Vignac, Igor Krawczuk, Antoine Siraudin.
MoFlow: Copyright (c) 2020 Chengxi Zang.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
