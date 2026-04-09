#!/usr/bin/env python

from argparse import ArgumentParser

from model import SELFIESmodel


def main(flags):
    print("\n----- Fine-tuning SELFIES LSTM model -----\n")
    if not flags.dataset:
        raise ValueError("Please specify the dataset for fine tuning!")

    run = flags.name if flags.name else flags.dataset.split("/")[-1].split(".")[0]

    model = SELFIESmodel(
        dataset=flags.dataset,
        num_epochs=flags.train,
        run_name=run,
        reinforce=bool(flags.reinforce),
        batch_size=flags.batch,
        validation=flags.val,
        mw_filter=flags.mw_filter.split(",") if flags.mw_filter else None,
        sample_after=flags.after,
        lr=flags.lr,
        reference=flags.reference,
        num_reinforce=flags.num_reinforce,
        workers=flags.workers,
        seed=flags.seed,
        reward=flags.reward,
        reward_weight=flags.reward_weight,
    )

    model.load_data(preprocess=flags.preprocess, stereochem=flags.stereo, augment=flags.augment)

    vocab_path = flags.model + "vocab.json"
    model.load_vocab(vocab_path)
    model._tokenize_all()  # re-tokenize with pretrained vocab
    model.save_vocab()     # save vocab to new checkpoint dir

    model.load_model_from_file(checkpoint_dir=flags.model, epoch=flags.epoch)
    model.model.layers[1].trainable = False  # freeze first LSTM layer
    model.model.compile(loss="categorical_crossentropy", optimizer=model.model.optimizer, metrics=["accuracy"])

    print("Pre-trained model loaded, finetuning...")

    model.train_model(n_sample=flags.sample)

    if flags.reward and flags.pg_steps > 0:
        print(f"\nRunning {flags.pg_steps} property-guided REINFORCE steps (reward={flags.reward})...")
        for step in range(flags.pg_steps):
            loss = model.property_guided_step(n_sample=flags.pg_sample, temp=flags.temp)
            if (step + 1) % 10 == 0:
                print(f"  PG step {step + 1}/{flags.pg_steps}, loss: {loss:.4f}")

    valid_mols = model.sample_points(flags.sample, flags.temp)
    with open("./generated/" + run + "_finetuned.csv", "a") as f:
        f.write("\n".join(valid_mols))
    print(f"Valid:\t{len(valid_mols)}/{flags.sample}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="path (folder) of the pretrained model")
    parser.add_argument("--dataset", required=True, type=str, help="dataset for fine tuning")
    parser.add_argument("--name", type=str, default="transferlearn", help="run name for output files")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=19, help="epoch to load")
    parser.add_argument("--train", type=int, default=5, help="number of epochs to fine tune")
    parser.add_argument("--sample", type=int, default=25, help="number of points to sample during and after training")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--after", type=int, default=1, help="sample after how many epochs (0 = no sampling)")
    parser.add_argument(
        "--augment", type=int, default=5, help="SELFIES augmentation factor via SMILES randomisation [1-n]"
    )
    parser.add_argument("--batch", type=int, default=64, help="batch size for finetuning")
    parser.add_argument("--preprocess", dest="preprocess", action="store_true", default=True)
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--stereo", type=int, default=0, help="whether stereochemistry should be included [0, 1]")
    parser.add_argument("--reinforce", dest="reinforce", action="store_true", default=True)
    parser.add_argument("--no-reinforce", dest="reinforce", action="store_false")
    parser.add_argument(
        "--num_reinforce", type=int, default=3, help="number of generated compounds to add back to training"
    )
    parser.add_argument(
        "--mw_filter", type=str, default="250,400", help="MW thresholds for reinforcing molecules (e.g. '250,400')"
    )
    parser.add_argument(
        "--reference", type=str, default="", help="reference molecule (SMILES) for reinforcement similarity"
    )
    parser.add_argument("--val", type=float, default=0.0, help="fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--workers", type=int, default=-1, help="number of threads for the data generator")
    parser.add_argument(
        "--reward", type=str, default=None, help="reward function for property-guided training (qed, logp, mw, tpsa)"
    )
    parser.add_argument("--reward_weight", type=float, default=0.1, help="weight of reward-guided loss")
    parser.add_argument(
        "--pg_steps", type=int, default=0, help="number of REINFORCE policy-gradient steps after fine-tuning"
    )
    parser.add_argument("--pg_sample", type=int, default=64, help="number of molecules to sample per PG step")
    args = parser.parse_args()

    if not args.model.endswith("/"):
        args.model += "/"

    main(args)
