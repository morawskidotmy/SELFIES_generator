import numpy as np
import tensorflow as tf
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem.QED import qed


def qed_reward(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return qed(mol)


def logp_reward(smiles, target=2.5, sigma=1.0):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    logp = Descriptors.MolLogP(mol)
    return float(np.exp(-((logp - target) ** 2) / (2 * sigma**2)))


def mw_reward(smiles, target=350.0, sigma=50.0):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    return float(np.exp(-((mw - target) ** 2) / (2 * sigma**2)))


def tpsa_reward(smiles, target=80.0, sigma=20.0):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    tpsa = Descriptors.TPSA(mol)
    return float(np.exp(-((tpsa - target) ** 2) / (2 * sigma**2)))


def multi_reward(smiles, reward_fns, weights=None):
    if weights is None:
        weights = [1.0] * len(reward_fns)
    total_w = sum(weights)
    score = sum(w * fn(smiles) for w, fn in zip(weights, reward_fns))
    return score / total_w


REWARD_REGISTRY = {
    "qed": qed_reward,
    "logp": logp_reward,
    "mw": mw_reward,
    "tpsa": tpsa_reward,
}


def get_reward_fn(name):
    fn = REWARD_REGISTRY.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown reward '{name}'. Available: {list(REWARD_REGISTRY.keys())}")
    return fn


class PolicyGradientLoss(tf.keras.losses.Loss):
    def __init__(self, baseline=0.5, **kwargs):
        super().__init__(**kwargs)
        self.baseline = baseline

    def call(self, y_true, y_pred):
        xent = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=-1)
        return tf.reduce_mean(xent)


def reward_weighted_crossentropy(y_true, y_pred, rewards, baseline=0.5):
    xent = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=-1)  # (batch, seq_len)
    seq_loss = tf.reduce_mean(xent, axis=-1)  # (batch,)
    advantage = rewards - baseline
    return tf.reduce_mean(seq_loss * advantage)
