# SELFIES Generator

[![Ruff](https://github.com/morawskidotmy/selfies_generator/actions/workflows/ruff.yml/badge.svg)](https://github.com/morawskidotmy/selfies_generator/actions/workflows/ruff.yml)

SELFIES-based generative RNN for molecule *de novo* design with property-guided generation. This project is a fork of https://github.com/alexarnimueller/SMILES_generator.

## Introduction

This repository uses generative recurrent neural networks (RNN) with LSTM cells to learn molecular structures represented as **SELFIES** (SELF-referencIng Embedded Strings). SELFIES provide a 100% robust molecular string representation — every SELFIES string decodes to a valid molecule, eliminating the invalid-molecule problem inherent to SMILES.

After training, the model can generate novel molecules. An optional **property-guided generation** mode (REINFORCE) steers sampling towards compounds with desired properties (e.g. high QED, target LogP, MW, TPSA).

## Installation

```bash
git clone https://github.com/morawskidotmy/selfies_generator.git
cd selfies_generator
uv sync            # install all dependencies
uv sync --extra dev   # include ruff for linting
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Usage

### Training

```bash
uv run python train.py --dataset data/chembl24_10uM_20-100.csv --name test --train 20 \
    --lr 0.005 --batch 512 --after 2 --sample 100 --augment 5 --preprocess --stereo 1 \
    --val 0.1 --seed 42
```

**With property-guided reward logging** (logs mean QED each sampling round):

```bash
uv run python train.py --dataset data/chembl24_10uM_20-100.csv --name test_qed \
    --train 20 --reward qed
```

### Sampling

```bash
uv run python sample.py --model checkpoint/test/ --out generated/test_sampled.csv \
    --epoch 9 --num 1000 --temp 1.0 --seed 42
```

### Fine-tuning

```bash
uv run python finetune.py --model checkpoint/test/ --dataset data/actives.csv \
    --name test-finetune --lr 0.005 --epoch 19 --train 20 --sample 100 --temp 1.0 \
    --after 1 --augment 10 --batch 16 --preprocess --stereo 1 --val 0.0 --seed 42
```

**With property-guided REINFORCE steps** after fine-tuning:

```bash
uv run python finetune.py --model checkpoint/test/ --dataset data/actives.csv \
    --name test-qed-ft --reward qed --pg_steps 100 --pg_sample 64
```

### Analysis

```bash
uv run python analyze.py --generated generated/test_sampled.csv \
    --reference data/chembl24_10uM_20-100.csv --name test --n 3 --fingerprint ECFP4
```

## Property-Guided Generation

Available reward functions (via `--reward`):

| Name   | Description                              | Range |
|--------|------------------------------------------|-------|
| `qed`  | Quantitative Estimate of Drug-likeness   | [0,1] |
| `logp` | Gaussian reward centred on target LogP   | [0,1] |
| `mw`   | Gaussian reward centred on target MW     | [0,1] |
| `tpsa` | Gaussian reward centred on target TPSA   | [0,1] |

Custom reward functions can be added in `losses.py` and registered in `REWARD_REGISTRY`.

## References

- Krenn, M., Häse, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A. (2020) Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation. *Machine Learning: Science and Technology* 1, 045024.
- Gupta, A., Müller, A. T., Huisman, B. J. H., Fuchs, J. A., Schneider, P. and Schneider, G. (2018) Generative recurrent networks for de novo drug design. *Mol. Inf.* 37, 1700111.
