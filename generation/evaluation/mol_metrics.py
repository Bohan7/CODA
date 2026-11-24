"""Metrics of generated molecule sets.

For every generated SMILES file this reports
  validity      fraction of SMILES that RDKit parses
  FCD           Fréchet ChemNet Distance to the training molecules
  E[p(y|G)]     mean confidence of the time-dependent classifier in the conditioning label
                (read from `<set>_confidences.pkl`, written by the sampler)

    python evaluation/mol_metrics.py --train_smiles data/goodhiv_train_smiles.txt samples/hiv_scaffold/*.txt
"""
import argparse
import os
import pickle

import numpy as np
import torch
from fcd_torch import FCD
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def canonical(smiles):
    mols = (Chem.MolFromSmiles(s) for s in smiles)
    return [Chem.MolToSmiles(m) for m in mols if m is not None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sets', nargs='+', help='generated SMILES files (.txt)')
    parser.add_argument('--train_smiles', required=True, help='training SMILES, one per line')
    parser.add_argument('--n', type=int, default=None, help='subsample this many molecules per set (default: all)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    def subsample(items):
        if args.n is None or len(items) <= args.n:
            return items
        return [items[i] for i in rng.choice(len(items), args.n, replace=False)]

    fcd = FCD(device='cuda' if torch.cuda.is_available() else 'cpu', n_jobs=4)
    with open(args.train_smiles) as f:
        train = canonical([line.strip() for line in f if line.strip()])
    ref_stats = fcd.precalc(subsample(train))

    print(f'{"set":80s} {"n":>7s} {"valid":>7s} {"FCD":>8s} {"E[p(y|G)]":>10s}')
    for path in args.sets:
        with open(path) as f:
            smiles = [line.strip() for line in f if line.strip()]
        valid = canonical(smiles)
        score = fcd(gen=subsample(valid), pref=ref_stats) if len(valid) > 10 else float('nan')
        conf_path = path[:-4] + '_confidences.pkl'
        conf = float('nan')
        if os.path.exists(conf_path):
            with open(conf_path, 'rb') as f:
                conf = float(np.mean(pickle.load(f)))
        print(f'{os.path.basename(path)[-80:]:80s} {len(smiles):7d} {len(valid) / max(len(smiles), 1):7.4f} '
              f'{score:8.3f} {conf:10.4f}', flush=True)
