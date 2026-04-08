#!/usr/bin/env python

from multiprocessing import Process, Queue, cpu_count
from time import time

import numpy as np
import selfies as sf
from rdkit.Chem import (
    CanonSmiles,
    MolFromSmiles,
    MolToInchiKey,
    MolToSmiles,
    RenumberAtoms,
    ReplaceCore,
    ReplaceSidechains,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from descriptors import cats_descriptor, numpy_fps, numpy_maccs, parallel_pairwise_similarities

START_TOKEN = "^"
END_TOKEN = "$"
PAD_TOKEN = " "


def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles(selfies_str):
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return None


def split_selfies(selfies_str):
    return list(sf.split_selfies(selfies_str))


def build_vocab(selfies_list):
    token_set = set()
    for s in selfies_list:
        if s is not None:
            token_set.update(sf.split_selfies(s))
    tokens = [START_TOKEN, END_TOKEN, PAD_TOKEN] + sorted(token_set)
    indices_token = {str(i): t for i, t in enumerate(tokens)}
    token_indices = {t: str(i) for i, t in enumerate(tokens)}
    return indices_token, token_indices


def tokenizer(text=None, mode="default"):
    if mode == "generate" and text is not None:
        return build_vocab(text)
    raise NotImplementedError(
        "The 'default' fixed tokenizer is removed for SELFIES. Use mode='generate' and pass your SELFIES list."
    )


def tokenize_selfies_string(selfies_str):
    tokens = []
    i = 0
    while i < len(selfies_str):
        ch = selfies_str[i]
        if ch == "[":
            end = selfies_str.index("]", i) + 1
            tokens.append(selfies_str[i:end])
            i = end
        else:
            tokens.append(ch)
            i += 1
    return tokens


def tokenize_to_indices(token_list, token_indices):
    return [int(token_indices[t]) for t in token_list]


def tokenize_molecules(token_lists, token_indices):
    result = []
    for tokens in token_lists:
        result.append([int(token_indices[t]) for t in tokens])
    return np.array(result)


def pad_token_seqs(token_seqs, pad_token=PAD_TOKEN, max_len=0):
    if max_len == 0:
        max_len = max(len(s) for s in token_seqs)
    return [s + [pad_token] * (max_len - len(s)) for s in token_seqs]


def one_hot_encode(token_lists, n_chars):
    token_lists = np.asarray(token_lists)
    output = np.zeros((*token_lists.shape, n_chars))
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            output[i, j, int(token)] = 1
    return output


def transform_temp(preds, temp):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def is_valid_mol(selfies_str, return_smiles=False):
    clean = selfies_str.replace(START_TOKEN, "").replace(END_TOKEN, "").replace(PAD_TOKEN, "").strip()
    smiles = selfies_to_smiles(clean)
    if smiles is None:
        return (False, None) if return_smiles else False
    try:
        canon = CanonSmiles(smiles, 1)
    except Exception:
        canon = None
    if canon is None:
        return (False, None) if return_smiles else False
    if return_smiles:
        return True, canon
    return True


def read_smiles_file(dataset):
    smls = []
    print(f"Reading {dataset}...")
    with open(dataset) as f:
        for line in f:
            s = line.strip()
            if s:
                smls.append(s)
    return smls


def randomize_smiles(smiles, num=10, isomeric=True):
    m = MolFromSmiles(smiles)
    if m is None:
        return []
    res = set()
    start = time()
    while len(res) < num and (time() - start) < 5:
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = RenumberAtoms(m, ans)
        res.add(MolToSmiles(nm, canonical=False, isomericSmiles=isomeric))
    return list(res)


def randomize_smileslist(smiles, num=10, isomeric=True):

    def _one_random(smls, n, iso, q):
        res = []
        for s in smls:
            m = MolFromSmiles(s)
            if m is None or m.GetNumAtoms() <= 5:
                sel = smiles_to_selfies(s)
                if sel:
                    res.append(sel)
                continue
            r = set()
            start = time()
            while len(r) < n and (time() - start) < 5:
                ans = list(range(m.GetNumAtoms()))
                np.random.shuffle(ans)
                nm = RenumberAtoms(m, ans)
                smi = MolToSmiles(nm, canonical=False, isomericSmiles=iso)
                sel = smiles_to_selfies(smi)
                if sel:
                    r.add(sel)
            res.extend(r)
        q.put(res)

    queue = Queue()
    rslt = []
    for chunk in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=_one_random, args=(chunk, num, isomeric, queue))
        p.start()
    for _ in range(cpu_count()):
        rslt.extend(queue.get(timeout=30))
    return list(set(rslt))


def inchikey_from_smileslist(smiles):

    def _one_inchi(smls, q):
        res = []
        for s in smls:
            try:
                res.append(MolToInchiKey(MolFromSmiles(s)))
            except Exception:
                res.append(None)
        q.put(res)

    queue = Queue()
    rslt = []
    for chunk in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=_one_inchi, args=(chunk, queue))
        p.start()
    for _ in range(cpu_count()):
        rslt.extend(queue.get(timeout=10))
    return list(rslt)


def compare_inchikeys(target, reference):
    idx = [i for i, k in enumerate(target) if k not in reference]
    return [target[i] for i in idx], idx


def compare_mollists(smiles, reference, canonicalize=True):
    smiles = [s.replace(START_TOKEN, "").replace(END_TOKEN, "").strip() for s in smiles]
    reference = [s.replace(START_TOKEN, "").replace(END_TOKEN, "").strip() for s in reference]
    if canonicalize:
        mols = set(CanonSmiles(s, 1) for s in smiles if MolFromSmiles(s))
        refs = set(CanonSmiles(s, 1) for s in reference if MolFromSmiles(s))
    else:
        mols = set(smiles)
        refs = set(reference)
    return [m for m in mols if m not in refs]


def extract_murcko_scaffolds(mol):
    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        return MolToSmiles(core, isomericSmiles=True)
    except Exception:
        return ""


def extract_murcko_scaffolds_marked(mol, mark="[*]"):
    pos = range(0, 20)
    set_pos = ["[" + str(x) + "*]" for x in pos]
    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        tmp = ReplaceSidechains(m1, core)
        smi = MolToSmiles(tmp, isomericSmiles=True)
    except Exception:
        return ""
    for i in pos:
        smi = smi.replace(set_pos[i], mark)
    return smi


def extract_side_chains(mol, remove_duplicates=False, mark="[*]"):
    pos = range(0, 20)
    set_pos = ["[" + str(x) + "*]" for x in pos]
    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        side_chain = ReplaceCore(m1, core)
        smi = MolToSmiles(side_chain, isomericSmiles=True)
    except Exception:
        return []
    for i in pos:
        smi = smi.replace(set_pos[i], mark)
    if remove_duplicates:
        return list(set(smi.split(".")))
    return smi.split(".")


def get_most_similar(smiles, referencemol, n=10, desc="FCFP4", similarity="tanimoto"):
    if desc.upper() == "FCFP4":
        d_lib = numpy_fps([MolFromSmiles(s) for s in smiles], 2, True, 1024)
        d_ref = numpy_fps([MolFromSmiles(referencemol)], 2, True, 1024)
    elif desc.upper() == "MACCS":
        d_lib = numpy_maccs([MolFromSmiles(s) for s in smiles])
        d_ref = numpy_maccs([MolFromSmiles(referencemol)])
    elif desc.upper() == "CATS":
        d_lib = cats_descriptor([MolFromSmiles(s) for s in smiles])
        d_ref = cats_descriptor([MolFromSmiles(referencemol)])
    else:
        raise NotImplementedError("Only FCFP4, MACCS or CATS fingerprints are available!")
    sims = parallel_pairwise_similarities(d_lib, d_ref, similarity).flatten()
    if desc == "CATS":
        top_n = np.argsort(sims)[:n][::-1]
    else:
        top_n = np.argsort(sims)[-n:][::-1]
    return np.array(smiles)[top_n].flatten(), sims[top_n].flatten()
