import os, json, collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
from torch.nn.utils import rnn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import BertLMHeadModel, BertTokenizer
from recipe_prep import *

def elementwise_apply(fn, *args):
    """A hack to apply fn like nn.Embedding, F.log_softmax to PackedSequence"""
    return rnn.PackedSequence(
        fn(*[(arg.data if type(arg) == rnn.PackedSequence else arg)
             for arg in args]), args[0].batch_sizes)


class Encoder(pl.LightningModule):

    def __init__(self, n_embeddings, max_length, embedding_dim, num_transformer_layers, nheads):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nheads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.max_length=max_length

    def forward(self, inp, mask=None):
        output = self.embedding(inp)
        # output = output.view(-1, self.max_length*self.embedding.embedding_dim)
        # output = self.transformer(output, mask)
        output = output.transpose(0,1)
        output = self.transformer(output, src_key_padding_mask=mask)
        return output


class Decoder(pl.LightningModule):

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp):
        output = self.softmax(self.out(inp))
        return output


class Seq2Seq(pl.LightningModule):
    def __init__(self, n_tokens_input, n_tokens_output, input_max_length, output_max_length, embedding_dim=512,
                 num_transformer_layers=4,nheads=8):
        super().__init__()
        #TODO: use parameters for all the arguments
        self.encoder = Encoder(n_embeddings=n_tokens_input,max_length=input_max_length, embedding_dim=embedding_dim, num_transformer_layers=num_transformer_layers,nheads=nheads)
        self.decoder = Decoder(embedding_dim, n_tokens_output)
        self.n_tokens_output = n_tokens_output

    def forward(self, inp):
        # in lightning, forward defines the prediction/inference actions
        out = self.decoder(self.encoder(inp))
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        # y = F.one_hot(y.T.to(torch.int64), self.n_tokens_output).to(torch.float32)
        y=y.to(torch.long)
        y_hat = self.forward(x)
        y_hat = y_hat.view(y.shape[0], -1, y.shape[1])
        #self.log('train_loss', loss)
        return F.cross_entropy(y_hat,y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    max_length = 64
    output_vocab, id_types = unified_vocab()
    df = load_data_and_preprocess('../data/seq2seq_4335716.csv', output_vocab, max_size=max_length)
    input_ids =         torch.Tensor(np.vstack(df["input_ids"])).type(torch.int)
    output_ids =        torch.Tensor(np.vstack(df["output_ids"])).type(torch.int)
    input_attention =   torch.Tensor(np.vstack(df["input_attention"])).type(torch.bool)
    output_attention =  torch.Tensor(np.vstack(df["output_attention"])).type(torch.bool)


    # enc = Encoder(n_embeddings=len(output_vocab) ,max_length=max_length, embedding_dim=512, num_transformer_layers=4,nheads=8)
    # dec = Decoder(512, len(output_vocab))
    # ret= enc(output_ids, output_attention)
    # ret = dec(ret)
    # print (ret)
    dataset = TensorDataset(output_ids, output_ids)  # create your datset
    my_dataloader = DataLoader(dataset) # create your dataloader
    train, val = random_split(dataset, [int(output_ids.shape[0]*0.9), output_ids.shape[0]-int(output_ids.shape[0]*0.9)])

    seq2seq = Seq2Seq(len(output_vocab),len(output_vocab),max_length,max_length)
    trainer = pl.Trainer()
    trainer.fit(seq2seq, DataLoader(train), DataLoader(val))
