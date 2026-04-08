import os
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import selfies as sf
import tensorflow as tf
from rdkit.Chem import Descriptors, MolFromSmiles

from descriptors import cats_descriptor, parallel_pairwise_similarities
from generator import DataGenerator
from losses import get_reward_fn, reward_weighted_crossentropy
from preprocess import preprocess_smiles, smiles_list_to_selfies
from utils import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    build_vocab,
    compare_inchikeys,
    inchikey_from_smileslist,
    is_valid_mol,
    one_hot_encode,
    randomize_smileslist,
    read_smiles_file,
    smiles_to_selfies,
    tokenize_selfies_string,
    transform_temp,
)


class SELFIESmodel:
    def __init__(
        self,
        batch_size=128,
        dataset="data/default",
        num_epochs=25,
        lr=0.005,
        sample_after=1,
        temp=1.0,
        step=1,
        run_name="default",
        reference=None,
        reinforce=False,
        num_reinforce=3,
        mw_filter=None,
        workers=1,
        validation=0.2,
        seed=42,
        reward=None,
        reward_weight=0.1,
    ):
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))
        self.lr = lr
        self.dataset = dataset
        self.n_mols = 0
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sample_after = sample_after
        if self.sample_after == 0:
            self.sample_after = self.num_epochs + 1
        self.run_name = run_name
        self.checkpoint_dir = "./checkpoint/" + run_name + "/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.temp = temp
        self.reference = reference
        self.step = step
        self.reinforce = reinforce
        self.num_reinforce = num_reinforce
        self.validation = validation
        self.mw_filter = mw_filter
        self.n_chars = None
        self.molecules = None  # canonical SMILES (for reference / InChI)
        self.selfies = None  # SELFIES strings (working representation)
        self.val_mols = None
        self.train_mols = None
        self.token_indices = None
        self.indices_token = None
        self.token_seqs = None  # pre-tokenised int sequences
        self.smiles = None
        self.inchi = None
        self.model = None
        self.maxlen = None  # max token-level length
        self.reward_fn = get_reward_fn(reward) if reward else None
        self.reward_weight = reward_weight
        if workers == -1:
            self.workers = cpu_count()
            self.multi = True
        elif workers == 1:
            self.workers = 1
            self.multi = False
        else:
            self.workers = workers
            self.multi = True

    def load_data(self, preprocess=False, stereochem=1.0, augment=1):
        all_mols = read_smiles_file(self.dataset)
        if preprocess:
            all_mols = preprocess_smiles(all_mols, stereochem)
        self.molecules = all_mols
        self.smiles = all_mols
        print(f"{len(self.molecules)} molecules loaded from {self.dataset}...")

        print("Converting SMILES to SELFIES...")
        selfies_list = smiles_list_to_selfies(all_mols)
        print(f"  {len(selfies_list)} SELFIES strings obtained")

        token_lengths = [len(list(sf.split_selfies(s))) for s in selfies_list]
        self.maxlen = max(token_lengths) + 2  # +2 for ^ and $
        print(f"Maximal token-sequence length: {self.maxlen - 2}")

        print("Creating InChI keys...")
        self.inchi = inchikey_from_smileslist(all_mols)

        if augment > 1:
            print(f"Augmenting SELFIES {augment}-fold (via SMILES randomisation)...")
            augmented = randomize_smileslist(self.molecules, num=augment)
            print(f"  {len(augmented)} SELFIES strings generated for {len(self.molecules)} molecules")
            selfies_list = augmented
            token_lengths = [len(list(sf.split_selfies(s))) for s in selfies_list]
            self.maxlen = max(token_lengths) + 2

        self.selfies = selfies_list

        print("Building SELFIES vocabulary...")
        self.indices_token, self.token_indices = build_vocab(selfies_list)
        self.n_chars = len(self.indices_token)
        print(f"  Vocabulary size: {self.n_chars} tokens")

        self._tokenize_all()

        self.n_mols = len(self.token_seqs)
        self.val_mols, self.train_mols = np.split(
            np.random.choice(range(self.n_mols), self.n_mols, replace=False),
            [int(self.validation * self.n_mols)],
        )
        print(f"Using {len(self.train_mols)} examples for training and {len(self.val_mols)} for validation")

    def _tokenize_all(self):
        seqs = []
        for s in self.selfies:
            tokens = [START_TOKEN] + list(sf.split_selfies(s)) + [END_TOKEN]
            tokens += [PAD_TOKEN] * (self.maxlen - len(tokens))
            indices = [int(self.token_indices[t]) for t in tokens]
            seqs.append(indices)
        self.token_seqs = seqs

    def build_model(self):
        l_in = tf.keras.layers.Input(shape=(None, self.n_chars), name="Input")
        l_out = tf.keras.layers.LSTM(512, unit_forget_bias=True, return_sequences=True, name="LSTM_1")(l_in)
        l_out = tf.keras.layers.GaussianDropout(0.25, name="Dropout_1")(l_out)
        l_out = tf.keras.layers.LSTM(256, unit_forget_bias=True, return_sequences=True, name="LSTM_2")(l_out)
        l_out = tf.keras.layers.GaussianDropout(0.25, name="Dropout_2")(l_out)
        l_out = tf.keras.layers.BatchNormalization(name="BatchNorm")(l_out)
        l_out = tf.keras.layers.Dense(self.n_chars, activation="softmax", name="Dense")(l_out)
        self.model = tf.keras.models.Model(l_in, l_out)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )

    def train_model(self, n_sample=100):
        print("Training model...")
        log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.create_file_writer(log_dir)
        mol_file = open("./generated/" + self.run_name + "_generated.csv", "a")

        for i in range(self.num_epochs):
            print(f"\n------ ITERATION {i} ------")
            self.set_lr(i)
            print(f"\nCurrent learning rate: {float(self.model.optimizer.learning_rate):.5f}")

            chkpntr = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_dir + f"model_epoch_{i:02d}.keras", verbose=1
            )

            if self.validation:
                gen_train = DataGenerator(
                    self.token_seqs, self.train_mols, self.maxlen - 1, self.n_chars, self.step, self.batch_size
                )
                gen_val = DataGenerator(
                    self.token_seqs, self.val_mols, self.maxlen - 1, self.n_chars, self.step, self.batch_size
                )
                history = self.model.fit(
                    gen_train,
                    epochs=1,
                    validation_data=gen_val,
                    use_multiprocessing=self.multi,
                    workers=self.workers,
                    callbacks=[chkpntr],
                )
                with writer.as_default():
                    tf.summary.scalar("val_loss", history.history["val_loss"][-1], step=i)
            else:
                gen = DataGenerator(
                    self.token_seqs, range(self.n_mols), self.maxlen - 1, self.n_chars, self.step, self.batch_size
                )
                history = self.model.fit(
                    gen, epochs=1, use_multiprocessing=self.multi, workers=self.workers, callbacks=[chkpntr]
                )

            with writer.as_default():
                tf.summary.scalar("loss", history.history["loss"][-1], step=i)
                tf.summary.scalar("lr", float(self.model.optimizer.learning_rate), step=i)

            if (i + 1) % self.sample_after == 0:
                valid_mols = self.sample_points(n_sample, self.temp)
                n_valid = len(valid_mols)
                if n_valid:
                    print("Comparing novelty...")
                    inchi_valid = inchikey_from_smileslist(valid_mols)
                    inchi_novel, idx_novel = compare_inchikeys(inchi_valid, self.inchi)
                    novel = np.array(valid_mols)[idx_novel]
                    n_novel = float(len(set(inchi_novel))) / n_valid
                    mol_file.write(f"\n----- epoch {i} -----\n")
                    mol_file.write("\n".join(set(valid_mols)))

                    if self.reward_fn and len(valid_mols) > 0:
                        rewards = [self.reward_fn(s) for s in valid_mols]
                        mean_reward = np.mean(rewards)
                        print(f"Mean reward: {mean_reward:.4f}")
                        with writer.as_default():
                            tf.summary.scalar("mean_reward", mean_reward, step=i)
                else:
                    novel = []
                    n_novel = 0

                with writer.as_default():
                    tf.summary.scalar("valid", float(n_valid) / n_sample, step=i)
                    tf.summary.scalar("novel", n_novel, step=i)
                    tf.summary.scalar("unique_valid", len(set(valid_mols)), step=i)
                print(f"\nValid:\t{n_valid}/{n_sample}")
                print(f"Unique:\t{len(set(valid_mols))}")
                print(f"Novel:\t{len(novel)}\n")

                if self.reinforce:
                    self._reinforce_step(novel, n_sample)

        mol_file.close()

    def _reinforce_step(self, novel, n_sample):
        if len(novel) <= (n_sample / 5):
            return
        if self.mw_filter:
            mw = np.array([Descriptors.MolWt(MolFromSmiles(s)) if MolFromSmiles(s) else 0 for s in novel])
            mw_idx = np.where((int(self.mw_filter[0]) < mw) & (mw < int(self.mw_filter[1])))[0]
            novel = np.array(novel)[mw_idx]

        print("Calculating CATS similarities of novel generated molecules...")
        fp_novel = cats_descriptor([MolFromSmiles(s) for s in novel])
        if self.reference:
            fp_train = cats_descriptor([MolFromSmiles(self.reference)])
        else:
            fp_train = cats_descriptor([MolFromSmiles(s) for s in self.smiles])
        sims = parallel_pairwise_similarities(fp_novel, fp_train, metric="euclidean")
        top = sims[range(len(novel)), np.argsort(sims, axis=1)[:, 0, 0]].flatten()
        print(f"Adding top {self.num_reinforce} most similar but novel molecules to pool")

        top_smiles = novel[np.argsort(top)[: self.num_reinforce]]
        for smi in top_smiles:
            sel = smiles_to_selfies(smi)
            if sel is None:
                continue
            tokens = [START_TOKEN] + list(sf.split_selfies(sel)) + [END_TOKEN]
            if len(tokens) > self.maxlen:
                continue
            tokens += [PAD_TOKEN] * (self.maxlen - len(tokens))
            indices = [int(self.token_indices.get(t, 0)) for t in tokens]
            self.token_seqs.append(indices)

        order = np.random.permutation(len(self.token_seqs))
        self.token_seqs = [self.token_seqs[j] for j in order]

    def property_guided_step(self, n_sample=64, temp=1.0):
        if self.reward_fn is None:
            return

        valid_mols = self.sample_points(n_sample, temp)
        if len(valid_mols) == 0:
            return 0.0

        rewards = np.array([self.reward_fn(s) for s in valid_mols])
        baseline = np.mean(rewards)

        xs, ys = [], []
        for smi in valid_mols:
            sel = smiles_to_selfies(smi)
            if sel is None:
                continue
            tokens = [START_TOKEN] + list(sf.split_selfies(sel)) + [END_TOKEN]
            if len(tokens) > self.maxlen:
                tokens = tokens[: self.maxlen]
            tokens += [PAD_TOKEN] * (self.maxlen - len(tokens))
            indices = [int(self.token_indices.get(t, 0)) for t in tokens]
            xs.append(indices[:-1])
            ys.append(indices[1:])

        if len(xs) == 0:
            return 0.0

        x_ohe = one_hot_encode(xs, self.n_chars)
        y_ohe = one_hot_encode(ys, self.n_chars)
        x_tensor = tf.convert_to_tensor(x_ohe, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_ohe, dtype=tf.float32)
        r_tensor = tf.convert_to_tensor(rewards[: len(xs)], dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = self.model(x_tensor, training=True)
            loss = reward_weighted_crossentropy(y_tensor, preds, r_tensor, baseline)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss)

    def sample_points(self, n_sample=100, temp=1.0, prime_text=None, maxlen=200):
        if prime_text is None:
            prime_text = START_TOKEN
        valid_mols = []
        print(f"\n\n----- SAMPLING {n_sample} POINTS AT TEMP {temp:.2f} -----")
        for _ in range(n_sample):
            selfies_str = ""
            seed_tokens = tokenize_selfies_string(prime_text)
            seed_indices = [int(self.token_indices.get(t, 0)) for t in seed_tokens]

            end_idx = int(self.token_indices[END_TOKEN])
            step_count = 0
            while step_count < maxlen:
                x_seed = one_hot_encode([seed_indices], self.n_chars)
                preds = self.model.predict(x_seed, verbose=0)[0]
                next_idx = transform_temp(preds[-1, :], temp)
                if next_idx == end_idx:
                    break
                seed_indices.append(next_idx)
                step_count += 1

            generated_tokens = [self.indices_token[str(idx)] for idx in seed_indices]
            selfies_str = "".join(generated_tokens)

            val, s = is_valid_mol(selfies_str, True)
            if val:
                print(s)
                valid_mols.append(s)
        return valid_mols

    def sample(self, temp=1.0, prime_text=None, maxlen=200):
        if prime_text is None:
            prime_text = START_TOKEN
        seed_tokens = tokenize_selfies_string(prime_text)
        seed_indices = [int(self.token_indices.get(t, 0)) for t in seed_tokens]
        end_idx = int(self.token_indices[END_TOKEN])
        step_count = 0
        while step_count < maxlen:
            x_seed = one_hot_encode([seed_indices], self.n_chars)
            preds = self.model.predict(x_seed, verbose=0)[0]
            next_idx = transform_temp(preds[-1, :], temp)
            seed_indices.append(next_idx)
            if next_idx == end_idx:
                break
            step_count += 1
        return "".join(self.indices_token[str(idx)] for idx in seed_indices)

    def load_model_from_file(self, checkpoint_dir, epoch):
        model_file = checkpoint_dir + f"model_epoch_{epoch:02d}.keras"
        print("Loading model from file: " + model_file)
        self.model = tf.keras.models.load_model(model_file)

    def save_vocab(self, path=None):
        import json

        if path is None:
            path = self.checkpoint_dir + "vocab.json"
        with open(path, "w") as f:
            json.dump({"indices_token": self.indices_token, "token_indices": self.token_indices}, f)
        print(f"Vocabulary saved to {path}")

    def load_vocab(self, path):
        import json

        with open(path) as f:
            data = json.load(f)
        self.indices_token = data["indices_token"]
        self.token_indices = data["token_indices"]
        self.n_chars = len(self.indices_token)
        print(f"Vocabulary loaded ({self.n_chars} tokens)")

    def set_lr(self, epoch):
        new_lr = self.lr * np.power(0.5, np.floor((epoch + 1) / 5))
        self.model.optimizer.learning_rate.assign(new_lr)
