import numpy as np
import tensorflow as tf

from utils import one_hot_encode


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, token_seqs, ids, window, n_chars, step, batch_size, shuffle=True):
        self.token_seqs = token_seqs
        self.ids = ids
        self.window = window
        self.n_chars = n_chars
        self.step = step
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        return self.generate_xy([self.ids[k] for k in indexes])

    def __call__(self, *args, **kwargs):
        return self

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_xy(self, indexes):
        x, y = [], []
        for idx in indexes:
            seq = self.token_seqs[idx]  # list[int] — already tokenised
            for i in range(0, len(seq) - self.window, self.step):
                inp = seq[i : i + self.window]
                tgt = seq[i + 1 : i + self.window + 1]
                x.append(one_hot_encode([inp], self.n_chars)[0].tolist())
                y.append(one_hot_encode([tgt], self.n_chars)[0].tolist())
        return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)
