"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.legacy.data import Field, BucketIterator
import torch.utils.data as data
from functools import partial

from util.reverse_dataset import ReverseDataset

class DataLoaderReverse:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)
        self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)
        """
         elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) 
        """

        dataset = partial(ReverseDataset, 10, 16)
        self.train_data = dataset(50000)
        self.valid_data = dataset(1000)
        self.test_data = dataset(10000)
        return self.train_data, self.valid_data, self.test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        val_loader   = data.DataLoader(validate, batch_size=batch_size)
        test_loader  = data.DataLoader(test, batch_size=batch_size)
        print('dataset initializing done')
        return train_loader, val_loader, test_loader
