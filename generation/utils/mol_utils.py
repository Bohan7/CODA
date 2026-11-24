"""Dense tensors -> RDKit molecules, with the valency correction of MoFlow.

Adapted from MoFlow (https://github.com/calvin-zcx/moflow, MIT License) via GDSS.
"""
import re

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
BOND_DECODER = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def gen_mol(x, adj, atomic_num_list, largest_connected_comp=True):
    """Build one molecule per sample. x: B x N x (types + 1) one-hot, adj: B x 4 x N x N one-hot bonds.

    Returns the list of molecules (None where construction failed) and the number of samples that were
    already valid before the valency correction.
    """
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()
    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        mol, no_correct = correct_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
        num_no_correct += int(no_correct)
        mols.append(valid_mol_can_with_seg(mol, largest_connected_comp=largest_connected_comp))
    return mols, num_no_correct


def construct_mol(x, adj, atomic_num_list):
    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1                                # channels 0, 1, 2, 3 -> bond orders 1, 2, 3, 0 (0 = no bond)

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), BOND_DECODER[adj[start, end]])
            # add a formal charge to N, O, S atoms whose valence exceeds the default by one
            flag, atomid_valence = check_valency(mol)
            if not flag:
                assert len(atomid_valence) == 2
                idx, v = atomid_valence
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        e_sub = e[e.find('#'):]
        return False, list(map(int, re.findall(r'\d+', e_sub)))


def correct_mol(mol):
    """Lower the order of (or remove) the highest-order bond of every over-valent atom until valid."""
    no_correct, _ = check_valency(mol)
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        assert len(atomid_valence) == 2
        idx = atomid_valence[0]
        queue = [(b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in mol.GetAtomWithIdx(idx).GetBonds()]
        queue.sort(key=lambda tup: tup[1], reverse=True)
        if len(queue) > 0:
            start, end, t = queue[0][2], queue[0][3], queue[0][1] - 1
            mol.RemoveBond(start, end)
            if t >= 1:
                mol.AddBond(start, end, BOND_DECODER[t])
    return mol, no_correct


def valid_mol_can_with_seg(mol, largest_connected_comp=True):
    """Canonicalise; for a disconnected molecule keep the fragment with the longest SMILES."""
    if mol is None:
        return None
    sm = Chem.MolToSmiles(mol, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        return Chem.MolFromSmiles(vsm[0][0])
    return Chem.MolFromSmiles(sm)
