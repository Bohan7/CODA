"""Stable-pattern preservation of a generated set: E[p(y | G)] under the time-dependent classifier.

The sampler stores p(y | G_0) of every generated graph in `<set>_confidences.pkl`; this script reports its
mean and the fraction of graphs that pass the confidence filter used downstream.

    python evaluation/label_preservation.py samples/motif_basis/*.pkl
"""
import argparse
import os
import pickle

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sets', nargs='+', help='generated sets (.pkl or .txt)')
    parser.add_argument('--threshold', type=float, default=0.9)
    args = parser.parse_args()

    print(f'{"set":80s} {"n":>7s} {"E[p(y|G)]":>10s} {f">={args.threshold}":>8s}')
    for path in args.sets:
        if path.endswith(('_labels.pkl', '_confidences.pkl', '.partial.pkl')):    # companions matched by a glob
            continue
        with open(path[:-4] + '_confidences.pkl', 'rb') as f:
            conf = np.asarray(pickle.load(f), dtype=np.float64)
        print(f'{os.path.basename(path)[-80:]:80s} {len(conf):7d} {conf.mean():10.4f} {(conf >= args.threshold).mean():8.4f}')
