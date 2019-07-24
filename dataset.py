import numpy as np
import pandas as pd


class dataset():
    def __init__(self, batch_size, seq_len):
        # Read the column names
        with open('kddcup.names') as f:
            labels = f.readline()[:-2].split(',')
            names = [line.split(':')[0] for line in f]
            names.append('label')

        # The data values into a dataframe
        data = pd.read_csv('kddcup.data_10_percent_corrected', header=None)
        data.columns = names

        # Seperate the features and labels
        labels = data.label
        del data['label']

        # Convert labels to good(0) and bad(1) labels
        labels = labels.map({'normal.': 0}).fillna(1)

        # Convert categorical variables to one-hot
        data = pd.get_dummies(data)

        self.data, self.labels = data.values, labels.values
        self.length = len(data)
        self.batch_size, self.seq_len = batch_size, seq_len
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.length:
            raise StopIteration

        idx = np.random.choice(self.length - self.seq_len, self.batch_size)

        batch = [self.data[i:i + self.seq_len] for i in idx]
        batch = np.transpose(batch, (1, 0, 2))

        labels = [self.labels[i:i + self.seq_len] for i in idx]
        labels = np.transpose(labels, (1, 0))
        labels = np.expand_dims(labels, 2)

        self.counter += 1

        return batch, labels