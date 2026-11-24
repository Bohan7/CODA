"""Convert GOOD-HIV into the generator's training files `<out_dir>/<name>_{train,valid,test}_kekulized.npz`.

    python data/prepare_good_hiv.py --domain scaffold --good_dir ../data/GOOD     # -> goodhiv_*.npz
    python data/prepare_good_hiv.py --domain size     --good_dir ../data/GOOD     # -> goodhivsize_*.npz

Each file stores three arrays: atomic numbers (N x max_atoms, 0 = padding), bond channels
(N x 4 x max_atoms x max_atoms: single, double, triple, aromatic) of the kekulised molecule, and the label.
Molecules with more than `max_atoms` heavy atoms are skipped (50 for the scaffold split, 70 for the size
split). The training SMILES are also written to `<name>_train_smiles.txt` (reference set for FCD).
The molecule featurisation follows MoFlow / chainer-chemistry (MIT License).
"""
import argparse
import os
import sys

import numpy as np
from rdkit import Chem
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'classification'))

BOND_CHANNEL = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
MAX_ATOMS = {'scaffold': 50, 'size': 70}
NAME = {'scaffold': 'goodhiv', 'size': 'goodhivsize'}


def featurize(smiles, max_atoms):
    """Canonicalise, kekulise and return (atomic numbers, bond channels)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True))
    Chem.Kekulize(mol)
    n = mol.GetNumAtoms()
    if n > max_atoms:
        return None
    atoms = np.zeros(max_atoms, dtype=np.int32)
    atoms[:n] = [a.GetAtomicNum() for a in mol.GetAtoms()]
    bonds = np.zeros((4, max_atoms, max_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[BOND_CHANNEL[bond.GetBondType()], i, j] = 1.0
        bonds[BOND_CHANNEL[bond.GetBondType()], j, i] = 1.0
    return atoms, bonds


def convert(split, max_atoms):
    atoms, bonds, labels, smiles = [], [], [], []
    for data in DataLoader(split, batch_size=1, shuffle=False):
        if data.x.shape[0] > max_atoms:
            continue
        try:
            feats = featurize(data.smiles[0], max_atoms)
        except Exception:                       # molecules RDKit cannot kekulise
            feats = None
        if feats is None:
            continue
        atoms.append(feats[0])
        bonds.append(feats[1])
        labels.append([int(data.y[0][0].item())])
        smiles.append(data.smiles[0])
    return np.asarray(atoms), np.asarray(bonds), np.asarray(labels), smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=['scaffold', 'size'])
    parser.add_argument('--good_dir', default='../data/GOOD')
    parser.add_argument('--out_dir', default='data')
    args = parser.parse_args()

    from GOOD.data.good_datasets.good_hiv import GOODHIV
    dataset, _ = GOODHIV.load(args.good_dir, domain=args.domain, shift='covariate', generate=False)
    os.makedirs(args.out_dir, exist_ok=True)
    for split, tag in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        atoms, bonds, labels, smiles = convert(dataset[split], MAX_ATOMS[args.domain])
        path = os.path.join(args.out_dir, f'{NAME[args.domain]}_{tag}_kekulized.npz')
        np.savez(path, atoms, bonds, labels)
        print(f'{split}: kept {len(atoms)} of {len(dataset[split])} molecules -> {path}')
        if split == 'train':
            with open(os.path.join(args.out_dir, f'{NAME[args.domain]}_train_smiles.txt'), 'w') as f:
                f.writelines(s + '\n' for s in smiles)
