#!/usr/bin/env python

from argparse import ArgumentParser

import tensorflow as tf

from model import SELFIESmodel
from utils import smiles_to_selfies


def main(flags):
    print("\n----- Sampling from SELFIES LSTM model -----\n")
    model = SELFIESmodel(dataset=flags.model, seed=flags.seed)

    vocab_path = flags.model + "vocab.json"
    model.load_vocab(vocab_path)
    model.load_model_from_file(flags.model, flags.epoch)

    prime = flags.frag if flags.frag.startswith("^") else "^" + flags.frag
    print(f"Starting token(s): {prime}")
    valid_mols = model.sample_points(n_sample=flags.num, temp=flags.temp, prime_text=prime)

    with open(flags.out, "w") as f:
        if flags.out.endswith(".sfi"):
            for smi in set(valid_mols):
                sel = smiles_to_selfies(smi)
                if sel:
                    f.write(f"{sel}\t{smi}\n")
        else:
            f.write("\n".join(set(valid_mols)))
    print(f"Valid: {len(valid_mols)}/{flags.num}")
    print(f"Unique: {len(set(valid_mols))}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="checkpoint/chembl24/", help="model path within checkpoint directory"
    )
    parser.add_argument("--out", type=str, default="generated/chembl24_sampled.csv", help="output file for molecules")
    parser.add_argument("--epoch", type=int, default=14, help="epoch to load")
    parser.add_argument("--num", type=int, default=100, help="number of points to sample")
    parser.add_argument("--temp", type=float, default=0.9, help="sampling temperature")
    parser.add_argument(
        "--frag", type=str, default="^", help="Fragment to grow SELFIES from. default: start character '^'"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)
