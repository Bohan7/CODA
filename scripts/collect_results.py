"""Summarise result files written by `classification/train.py --out` (mean ± std over seeds).

    python scripts/collect_results.py results/            # one row per sub-directory / file group
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()

    groups = defaultdict(dict)                           # group -> seed -> result
    for path in sorted(glob.glob(os.path.join(args.root, '**', '*.json'), recursive=True)):
        with open(path) as f:
            for r in json.load(f)['results']:
                groups[os.path.relpath(os.path.dirname(path), args.root)][r['seed']] = r

    print(f'{"run":45s} {"seeds":>5s} {"test @ OOD-val":>16s} {"test @ ID-val":>16s}')
    for name, by_seed in sorted(groups.items()):
        ood = [r['test@val'] for r in by_seed.values()]
        idv = [r['test@id_val'] for r in by_seed.values()]
        print(f'{name:45s} {len(ood):5d} {np.mean(ood):8.2f} ± {np.std(ood):5.2f} {np.mean(idv):8.2f} ± {np.std(idv):5.2f}')
