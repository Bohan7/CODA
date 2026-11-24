"""CODA sampling: generate a labelled, confidence-scored augmentation set with the guided reverse SDE."""
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from sde import load_sde
from solver import get_coda_sampler
from utils.data import ATOM_LIST, is_mol, train_node_flags
from utils.graph_utils import adjs_to_graphs, init_flags, init_random_flags, quantize, quantize_mol
from utils.config import to_plain
from utils.loader import load_ckpt, load_device, load_model_from_ckpt, load_seed
from utils.mol_utils import gen_mol


class Sampler:
    """Draws `sample.n_graphs` class-conditional graphs with exploration strength `sampler.ood` (lambda).

    Outputs, for an output prefix `<out>`:
      <out>.pkl               list of networkx graphs (generic datasets) or
      <out>.txt               one SMILES per line (molecules)
      <out>_labels.pkl        conditioning label of every generated graph
      <out>_confidences.pkl   p(y | G_0) under the time-dependent classifier, used as a filter downstream
    """

    def __init__(self, config, out):
        self.config = config
        self.out = out
        self.device = load_device()

    # ------------------------------------------------------------ setup
    def _setup(self):
        cfg = self.config
        if cfg.sample.n_graphs % cfg.sample.batch_size != 0:
            raise ValueError('sample.n_graphs must be a multiple of sample.batch_size')
        self.classifier = load_model_from_ckpt(load_ckpt(cfg.ckpt.classifier, self.device), self.device)
        ckpt = load_ckpt(cfg.ckpt.score, self.device)
        self._check_config(ckpt['config'])
        self.model = load_model_from_ckpt(ckpt, self.device, use_ema=cfg.sample.use_ema)
        self.train_flags = self._train_flags()

        self.max_node_num = cfg.data.max_node_num + cfg.sampler.ood_size
        shape_x = (cfg.sample.batch_size, self.max_node_num, cfg.data.max_feat_num)
        shape_adj = (cfg.sample.batch_size, self.max_node_num, self.max_node_num)
        self.sampling_fn = get_coda_sampler(
            load_sde(cfg.sde.x), load_sde(cfg.sde.adj), shape_x, shape_adj,
            predictor=cfg.sampler.predictor, corrector=cfg.sampler.corrector,
            snr=cfg.sampler.snr, scale_eps=cfg.sampler.scale_eps,
            weight_x=cfg.sampler.weight_x_classifier, weight_adj=cfg.sampler.weight_adj_classifier,
            lam=cfg.sampler.ood,
            denoise=cfg.sample.noise_removal, eps=cfg.sample.eps, device=self.device)

    def _check_config(self, trained):
        """The config must describe the data and the SDEs the score network was trained with."""
        cfg = self.config
        pairs = [(f'data.{k}', cfg.data[k], trained.data[k]) for k in ('data', 'max_node_num', 'max_feat_num')]
        pairs += [(f'sde.{part}.{k}', cfg.sde[part][k], trained.sde[part][k])
                  for part in ('x', 'adj') for k in ('type', 'beta_min', 'beta_max', 'num_scales')]
        wrong = [f'{name}: config {a!r}, checkpoint {b!r}' for name, a, b in pairs if a != b]
        if wrong:
            raise ValueError(f'{cfg.ckpt.score} was trained with different settings: ' + '; '.join(wrong))

    def _train_flags(self):
        """Node masks of the training graphs (empirical size prior).

        They are cached next to the score checkpoint as `train_node_flags.pt`, so sampling from a released
        checkpoint does not require the generator's training data.
        """
        path = os.path.join(os.path.dirname(self.config.ckpt.score), 'train_node_flags.pt')
        if os.path.exists(path):
            return torch.load(path, weights_only=True).to(torch.float32)
        flags = train_node_flags(self.config)
        torch.save(flags.to(torch.uint8), path)
        return flags

    def _labels(self):
        """Conditioning labels, arranged in class blocks with proportions `sample.class_ratio`."""
        n, ratio = self.config.sample.n_graphs, list(self.config.sample.class_ratio)
        if len(set(ratio)) == 1:                            # balanced classes
            counts = [math.ceil(n / len(ratio))] * len(ratio)
        else:                                               # e.g. GOOD-HIV: inactive : active = 24 : 1
            counts = [max(1, round(n * r / sum(ratio))) for r in ratio[1:]]
            counts = [n - sum(counts)] + counts
        labels = [c for c, k in enumerate(counts) for _ in range(k)]
        return torch.tensor(labels, dtype=torch.long)[:n]

    def _init_flags(self):
        cfg = self.config
        if cfg.sampler.ood_size != 0:                       # node-count prior Uniform{min_node_num, ..., max + ood_size}
            flags = init_random_flags(cfg.data.min_node_num, self.max_node_num, cfg.sample.batch_size)
        else:                                               # empirical node counts of the training graphs
            flags = init_flags(self.train_flags, cfg.sample.batch_size)
        return flags.to(self.device)

    # ------------------------------------------------------------ sampling
    def sample(self, labels=None):
        """Generate the set. `labels` overrides the conditioning labels (default: class blocks, `_labels`)."""
        cfg = self.config
        self._setup()
        load_seed(cfg.sample.seed)
        labels = self._labels() if labels is None else labels
        num_rounds = cfg.sample.n_graphs // cfg.sample.batch_size

        state = self._load_partial()
        for r in range(state['round'], num_rounds):
            t_start = time.time()
            y = labels[r * cfg.sample.batch_size:(r + 1) * cfg.sample.batch_size]
            flags = self._init_flags()
            x, adj = self.sampling_fn(self.model, self.classifier, flags, y)

            # confidence of the time-dependent classifier on the generated graph, p(y | G_0)
            self.classifier.eval()
            vec_t = torch.ones(adj.shape[0], device=adj.device) * cfg.sample.eps
            with torch.no_grad():
                _, _, logits = self.classifier(x, adj, vec_t.unsqueeze(-1), flags)
            confidence = F.softmax(logits, dim=-1)[range(adj.shape[0]), y].cpu()
            keep = confidence > 0.0                         # drops graphs whose confidence underflowed or is NaN
            state['x'].append(x[keep].cpu())
            state['adj'].append(adj[keep].cpu())
            state['labels'].append(y[keep])
            state['confidences'].append(confidence[keep])
            state['round'] = r + 1
            self._save_partial(state)
            print(f'Round {r} : {time.time() - t_start:.2f}s', flush=True)

        x, adj = torch.cat(state['x']), torch.cat(state['adj'])
        labels, confidences = torch.cat(state['labels']), torch.cat(state['confidences'])
        if is_mol(cfg):
            self._save_molecules(x, adj, labels.tolist(), confidences.tolist())
        else:
            self._save_graphs(x, adj, labels, confidences)
        if os.path.exists(self._partial_path()):
            os.remove(self._partial_path())

    # ------------------------------------------------------------ outputs
    def _save_graphs(self, x, adj, labels, confidences):
        with_features = self.config.data.get('node_features', 'degree') != 'degree'
        graphs = adjs_to_graphs(quantize(adj), labels, xs=x if with_features else None)
        self._dump(self.out + '.pkl', graphs)
        self._dump(self.out + '_labels.pkl', labels.tolist())
        self._dump(self.out + '_confidences.pkl', confidences.tolist())
        print(f'Saved {len(graphs)} graphs to {self.out}.pkl')

    def _save_molecules(self, x, adj, labels, confidences):
        bonds = quantize_mol(adj) - 1
        bonds[bonds == -1] = 3                               # bond orders 0, 1, 2, 3 -> channels 3, 0, 1, 2
        adj = F.one_hot(torch.tensor(bonds), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # append the "no atom" column

        mols, num_valid_wo_correction = gen_mol(x, adj, ATOM_LIST[self.config.data.data])
        smiles = [Chem.MolToSmiles(m) if m is not None else '' for m in mols]
        keep = [i for i, s in enumerate(smiles) if len(s)]   # drop failed / empty molecules, keeping alignment
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        with open(self.out + '.txt', 'w') as f:
            f.writelines(f'{smiles[i]}\n' for i in keep)
        self._dump(self.out + '_labels.pkl', [labels[i] for i in keep])
        self._dump(self.out + '_confidences.pkl', [confidences[i] for i in keep])
        print(f'Saved {len(keep)} molecules to {self.out}.txt '
              f'(valid without correction: {num_valid_wo_correction / max(len(mols), 1):.4f})')

    @staticmethod
    def _dump(path, obj):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------ per-round checkpointing
    def _partial_path(self):
        return self.out + '.partial.pkl'

    def _settings(self):
        """Everything that determines the generated set; a partial run is only resumed with identical settings."""
        cfg = self.config
        return to_plain({'data': cfg.data, 'sde': cfg.sde, 'ckpt': cfg.ckpt, 'sampler': cfg.sampler, 'sample': cfg.sample})

    def _load_partial(self):
        """Resume an interrupted run; the random states are restored so the result is unchanged."""
        if not os.path.exists(self._partial_path()):
            return {'round': 0, 'x': [], 'adj': [], 'labels': [], 'confidences': []}
        with open(self._partial_path(), 'rb') as f:
            state = pickle.load(f)
        if state.pop('settings') != self._settings():
            raise ValueError(f'{self._partial_path()} was written with different settings; '
                             'delete it or choose another --out')
        rng = state.pop('rng')
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng['cuda'] is not None:
            torch.cuda.set_rng_state(rng['cuda'])
        print(f"Resuming from round {state['round']}")
        return state

    def _save_partial(self, state):
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
               'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        tmp = self._partial_path() + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({**state, 'rng': rng, 'settings': self._settings()}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._partial_path())
