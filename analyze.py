#!/usr/bin/env python

from argparse import ArgumentParser

import tensorflow as tf
from rdkit.Chem import MolFromSmiles

from descriptors import cats_descriptor, numpy_fps, numpy_maccs, parallel_pairwise_similarities
from plotting import pca_plot, plot_top_n, sim_hist
from utils import compare_mollists


def main(flags):
    generated = [line.strip() for line in open(flags.generated)]
    reference = [line.strip() for line in open(flags.reference)]
    novels = compare_mollists(generated, reference, False)
    print(f"\n{len(generated)}\tgenerated molecules read")
    print(f"{len(reference)}\treference molecules read")
    print(f"{len(novels)}\tof the generated molecules are not in the reference set")

    try:
        from FCD import FCD_to_ref

        print("\nCalculating Fréchet ChEMBLNET Distance...")
        fcd = FCD_to_ref(generated, reference, n=min(len(generated), len(reference)))
        print(f"\nFréchet ChEMBLNET Distance to reference set:  {fcd:.4f}")
    except ImportError:
        print("\nFCD not installed, skipping Fréchet ChEMBLNET Distance. Install with: uv pip install FCD")

    print(f"\nCalculating {flags.fingerprint} similarities...")
    if flags.fingerprint == "ECFP4":
        similarity = "tanimoto"
        generated_fp = numpy_fps([MolFromSmiles(s) for s in generated], r=2, features=False, bits=1024)
        reference_fp = numpy_fps([MolFromSmiles(s) for s in reference], r=2, features=False, bits=1024)
    elif flags.fingerprint == "MACCS":
        similarity = "tanimoto"
        generated_fp = numpy_maccs([MolFromSmiles(s) for s in generated])
        reference_fp = numpy_maccs([MolFromSmiles(s) for s in reference])
    elif flags.fingerprint == "CATS":
        similarity = "euclidean"
        generated_fp = cats_descriptor([MolFromSmiles(s) for s in generated])
        reference_fp = cats_descriptor([MolFromSmiles(s) for s in reference])
    else:
        raise NotImplementedError('Only "MACCS", "CATS" or "ECFP4" are available as fingerprints!')

    sims = parallel_pairwise_similarities(generated_fp, reference_fp, similarity)
    sim_hist(sims.reshape(-1, 1), filename=f"./plots/{flags.name}_sim_hist.pdf")
    pca_plot(data=generated_fp, reference=reference_fp, filename=f"./plots/{flags.name}_pca.png")
    plot_top_n(
        smiles=novels,
        ref_smiles=reference,
        n=flags.num,
        fp=flags.fingerprint,
        sim=similarity,
        filename=f"./plots/{flags.name}_similar_mols.png",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--generated",
        type=str,
        default="generated/combined_data_10k_sampled.csv",
        help="the sampled molecules (SMILES, one per line)",
    )
    parser.add_argument(
        "--reference", type=str, default="data/combined_data.csv", help="the reference molecules (SMILES, one per line)"
    )
    parser.add_argument("--name", type=str, default="analyze_", help="name prepended to output filenames")
    parser.add_argument("--num", type=int, default=3, help="number of most similar molecules to return per reference")
    parser.add_argument("--fingerprint", type=str, default="ECFP4", help="fingerprint type: MACCS, ECFP4, or CATS")
    args = parser.parse_args()

    with tf.device("/GPU:0"):
        main(args)
