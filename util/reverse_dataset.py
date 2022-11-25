import torch.utils.data as data
import numpy as np
import torch

class ReverseDataset(data.Dataset):

    def __init__(self, num_categories, seq_len, n_samples):
        '''
            num_categories: range of the numbers; [0, num_categories-1]
            n_samples: number of sequences
            seq_len: length of the individual sequences
        '''
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.n_samples = n_samples

        self.data = np.random.randint(self.num_categories, size=(self.n_samples, self.seq_len))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = np.flip(inp_data, axis=-1)
        inp_data = torch.from_numpy(inp_data.copy())
        labels = torch.from_numpy(labels.copy())

        return inp_data, labels
