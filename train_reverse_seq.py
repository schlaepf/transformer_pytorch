# credit: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Sequence-to-Sequence

import torch
import time
import math
from torch import nn, optim
import torch.utils.data as data
from functools import partial
from torch.optim import Adam
from util.epoch_timer import epoch_time

from conf import *
from models.model.transformer import Transformer
from util.reverse_dataset import ReverseDataset
from data_reverse import *


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


""" dataset = partial(ReverseDataset, 10, 16)
train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader   = data.DataLoader(dataset(1000), batch_size=128)
test_loader  = data.DataLoader(dataset(10000), batch_size=128) """
# let's look at the first element of the dataset
#inp_data, labels = train_loader.dataset[0]
#print("Input data:", inp_data)
#print("Labels:    ", labels)

#enc_voc_size = len(train_loader.source.vocab)
#dec_voc_size = len(train_loader.target.vocab)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=factor,
                                                    patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        X = batch.src
        y = batch.trg

        optimizer.zero_grad()
        y_pred = model(X, y[:, :-1])
        y_pred_reshape = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y[:, 1:].contiguous().view(-1)

        loss = criterion(y_pred_reshape, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def run(total_epoch, best_loss):
    train_losses, test_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
