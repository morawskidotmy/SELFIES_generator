#!/usr/bin/env python

from multiprocessing import Process, Queue, cpu_count

import numpy as np
from rdkit.Chem import CanonSmiles

from utils import smiles_to_selfies


def keep_longest(smls):
    out = []
    for s in smls:
        if "." in s:
            f = s.split(".")
            out.append(f[np.argmax([len(m) for m in f])])
        else:
            out.append(s)
    return out


def harmonize_sc(mols):
    out = []
    for mol in mols:
        pairs = [
            ("[N](=O)[O-]", "[N+](=O)[O-]"),
            ("[O-][N](=O)", "[O-][N+](=O)"),
        ]
        for b, a in pairs:
            mol = mol.replace(b, a)
        out.append(mol)
    return out


def preprocess_smiles(smiles, stereochem=1):

    def process(s, q):
        smls = keep_longest(s)
        smls = harmonize_sc(smls)
        mols = []
        for smi in smls:
            try:
                mols.append(CanonSmiles(smi, stereochem))
            except Exception:
                print(f"Error! Cannot process SMILES string {smi}")
        q.put(mols)

    print("Preprocessing...")
    queue = Queue()
    for m in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=process, args=(m, queue))
        p.start()
    rslt = []
    for _ in range(cpu_count()):
        rslt.extend(queue.get(10))
    return np.random.choice(rslt, len(rslt), replace=False).tolist()


def smiles_list_to_selfies(smiles_list):
    selfies_list = []
    failed = 0
    for smi in smiles_list:
        sel = smiles_to_selfies(smi)
        if sel is not None:
            selfies_list.append(sel)
        else:
            failed += 1
    if failed:
        print(f"  {failed}/{len(smiles_list)} SMILES could not be converted to SELFIES")
    return selfies_list
