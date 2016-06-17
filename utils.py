import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, training_data_ratio=0.8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.training_data_ratio = training_data_ratio

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()
        counter = collections.Counter(data.split())
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data.split())))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        if self.num_batches < 50:
          print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = xdata.reshape(-1, self.batch_size, self.seq_length)
        self.y_batches = ydata.reshape(-1, self.batch_size, self.seq_length)

        # split data into training and validation, the default ratio is 0.8 and 0.2
        self.ntrain = np.floor(self.num_batches * self.training_data_ratio).astype('int')
        self.nvalidation = self.num_batches - self.ntrain #the rest goes to test (to ensure this adds up exactly)
        self.split_sizes = {'train': self.ntrain, 'validation': self.nvalidation}
        self.batch_index = {'train': 0, 'validation': 0}


    def next_batch(self, dataset_type):
        pointer = self.batch_index[dataset_type]
        if dataset_type == 'validation':
          pointer = pointer + self.ntrain # offset by train set size

        x, y = self.x_batches[pointer], self.y_batches[pointer]

        self.batch_index[dataset_type] += 1

        if self.batch_index[dataset_type] == self.split_sizes[dataset_type]:
            self.batch_index[dataset_type] = 0 # cycle around to the beginning

        return x, y

    def reset_batch_pointer(self, dataset_type):
        self.batch_index[dataset_type] = 0
