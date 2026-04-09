#!/usr/bin/env python

from argparse import ArgumentParser

import tensorflow as tf

from model import SELFIESmodel


def main(flags):
    print("\n----- Running SELFIES LSTM model -----\n")
    print("Initializing...")
    model = SELFIESmodel(
        batch_size=flags.batch,
        dataset=flags.dataset,
        num_epochs=flags.train,
        lr=flags.lr,
        run_name=flags.name,
        sample_after=flags.after,
        reinforce=flags.reinforce,
        validation=flags.val,
        reference=flags.ref,
        seed=flags.seed,
        reward=flags.reward,
        reward_weight=flags.reward_weight,
    )
    print("Loading data...")
    model.load_data(preprocess=flags.preprocess, stereochem=flags.stereo, augment=flags.augment)
    print("Saving vocabulary...")
    model.save_vocab()
    print("Building model...")
    model.build_model()
    print("Training...")
    model.train_model(n_sample=flags.sample)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/chembl24_10uM_20-100.csv",
        help="dataset file containing SMILES strings (one per line)",
    )
    parser.add_argument("--name", type=str, default="chembl24", help="run name for log and checkpoint files")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--after", type=int, default=2, help="sample after how many epochs")
    parser.add_argument("--sample", type=int, default=25, help="number of molecules to sample per sampling round")
    parser.add_argument("--train", type=int, default=20, help="number of epochs to train")
    parser.add_argument(
        "--augment", type=int, default=5, help="number of different SELFIES to generate via SMILES randomisation [1-n]"
    )
    parser.add_argument(
        "--preprocess", dest="preprocess", action="store_true", help="pre-process stereo chemistry/salts etc."
    )
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    parser.set_defaults(preprocess=False)
    parser.add_argument(
        "--stereo", type=int, default=0, help="whether stereo chemistry information should be included [0, 1]"
    )
    parser.add_argument(
        "--reinforce", action="store_true", default=False, help="add most similar but novel generated mols back to training"
    )
    parser.add_argument(
        "--ref", type=str, default=None, help="reference molecule (SMILES) for reinforcement similarity"
    )
    parser.add_argument("--val", type=float, default=0.1, help="fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--reward", type=str, default=None, help="reward function for property-guided training (qed, logp, mw, tpsa)"
    )
    parser.add_argument(
        "--reward_weight", type=float, default=0.1, help="weight of reward-guided loss relative to cross-entropy"
    )
    args = parser.parse_args()

    main(args)
