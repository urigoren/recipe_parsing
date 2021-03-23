import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
from torch.nn.utils import rnn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


def elementwise_apply(fn, *args):
    """A hack to apply fn like nn.Embedding, F.log_softmax to PackedSequence"""
    return rnn.PackedSequence(
        fn(*[(arg.data if type(arg) == rnn.PackedSequence else arg)
             for arg in args]), args[0].batch_sizes)


class EncoderRNN(pl.LightningModule):

    def __init__(self, hidden_size, embed_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inp, hidden):
        embedded = self.embedding(inp).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden


class DecoderRNN(pl.LightningModule):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        output = self.embedding(inp).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class Seq2Seq(pl.LightningModule):
    def __init__(self):
        super().__init__()
        n_tokens = 128
        hid_size = 12
        self.encoder = EncoderRNN(hid_size, n_tokens)
        self.decoder = DecoderRNN(hid_size, n_tokens)

    def forward(self, enc_out, enc_hid):
        # in lightning, forward defines the prediction/inference actions
        out, hid = self.encoder(enc_out, enc_hid)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.forward(x)
        x_hat = self.decoder(z)
        loss = nn.NLLLoss(x_hat, x)
        #self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


X = np.load("../data/dummy_X.npy")
Y = np.load("../data/dummy_Y.npy")
print (X.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=195)


tensor_x = torch.Tensor(X) # transform to torch tensor
tensor_y = torch.Tensor(Y)

dataset = TensorDataset(tensor_x,tensor_y) # create your datset
# my_dataloader = DataLoader(my_dataset) # create your dataloader
train, val = random_split(dataset, [9000, 1000])

seq2seq = Seq2Seq()
trainer = pl.Trainer()
trainer.fit(seq2seq, DataLoader(train), DataLoader(val))