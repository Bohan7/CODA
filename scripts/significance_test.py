"""Paired nonparametric bootstrap test between two methods evaluated on the same seeds.

    python scripts/significance_test.py results/coda.json results/baseline.json

Each file is the `--out` of `classification/train.py`. The p-value is the fraction of bootstrap resamples
(20000, fixed random state) whose mean paired difference is <= 0; when no resample falls below zero,
(0 + 1) / (20000 + 1) is reported.
"""
import argparse
import json

import numpy as np


def paired_bootstrap(a, b, n_boot=20000, seed=0):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.RandomState(seed)
    means = diff[rng.randint(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    count = int((means <= 0).sum())
    return diff.mean(), (count / n_boot if count > 0 else 1 / (n_boot + 1))


def load(path, key):
    with open(path) as f:
        return {r['seed']: r[key] for r in json.load(f)['results']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ours')
    parser.add_argument('baseline')
    parser.add_argument('--key', default='test@val', help='result field, e.g. test@val or test@id_val')
    args = parser.parse_args()

    ours, base = load(args.ours, args.key), load(args.baseline, args.key)
    seeds = sorted(set(ours) & set(base))                # pair by seed
    a, b = [ours[s] for s in seeds], [base[s] for s in seeds]
    delta, p = paired_bootstrap(a, b)
    print(f'{len(seeds)} shared seeds | ours {np.mean(a):.2f} ± {np.std(a):.2f} | baseline {np.mean(b):.2f} ± {np.std(b):.2f} '
          f'| mean difference {delta:+.2f} | p = {p:.4g}')
