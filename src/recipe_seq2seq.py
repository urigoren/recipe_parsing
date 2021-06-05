import os, json, collections
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
from torch.nn.utils import rnn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import BertLMHeadModel, BertTokenizer

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

    def forward(self, inp, mask):
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
    def __init__(self, input_vocab, output_vocab, input_max_length, output_max_length, embedding_dim=512,
                 num_transformer_layers=4,nheads=8):
        super().__init__()
        n_tokens = 128
        hid_size = 12
        #TODO: use parameters for all the arguments
        self.encoder = Encoder(n_embeddings=len(output_vocab),max_length=input_max_length, embedding_dim=embedding_dim, num_transformer_layers=num_transformer_layers,nheads=nheads)
        self.decoder = Decoder(hid_size, n_tokens)

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


def unified_vocab():
    output_vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "PUT": 2,
        "REMOVE": 3,
        "USE": 4,
        "STOP_USING": 5,
        "CHEF_CHECK": 6,
        "CHEF_DO": 7,
        "MOVE_CONTENTS": 8,
    }
    id_types = {'commands': [2, 3, 4, 5, 6, 7, 8], 'resources': [], 'args': []}
    k = len(output_vocab)
    with open("../data/res2idx.json", 'r') as f:
        for w, i in json.load(f).items():
            output_vocab[w] = k
            id_types['resources'].append(k)
            k += 1
    with open("../data/arg2idx.json", 'r') as f:
        for w, i in json.load(f).items():
            #         output_vocab[w] = k
            output_vocab[w.replace('-', '_')] = k
            id_types['args'].append(k)
            k += 1

    output_vocab = {w: i for i, w in enumerate(output_vocab)}
    return output_vocab, id_types


def load_data_and_preprocess(csv_file, output_vocab, max_size=128):
    def output_tokenize(s):
        ret = [1]
        for w in s.split():
            ret.append(output_vocab[w])
        ret.append(0)
        return pad(ret)

    def pad(lst):
        lst = np.array(lst)
        lst = np.pad(lst, (0,max_size-len(lst)))
        return lst
    input_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(csv_file)
    df["output_attention"] = df["output_seq"].apply(lambda s: pad(np.ones(len(s.split()))).astype('int'))
    df["output_ids"] = df["output_seq"].apply(output_tokenize)
    df["input_ids"] = df["input_seq"].apply(lambda x: pad(input_tokenizer(x, add_special_tokens=True).input_ids))
    df["input_attention"] = df["input_seq"].apply(
        lambda x: pad(input_tokenizer(x, add_special_tokens=True).attention_mask))
    return df



if __name__ == "__main__":
    max_length = 64
    output_vocab, id_types = unified_vocab()
    df = load_data_and_preprocess('../data/seq2seq_4335716.csv', output_vocab, max_size=max_length)
    input_ids =         torch.Tensor(np.vstack(df["input_ids"])).type(torch.int)
    output_ids =        torch.Tensor(np.vstack(df["output_ids"])).type(torch.int)
    input_attention =   torch.Tensor(np.vstack(df["input_attention"])).type(torch.bool)
    output_attention =  torch.Tensor(np.vstack(df["output_attention"])).type(torch.bool)


    enc = Encoder(n_embeddings=len(output_vocab) ,max_length=max_length, embedding_dim=512, num_transformer_layers=4,nheads=8)
    # ret= enc(output_ids, None)
    ret= enc(output_ids, output_attention)
    print (ret)
    print("="*88)
    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # my_dataloader = DataLoader(my_dataset) # create your dataloader
    train, val = random_split(dataset, [int(X.shape[0]*0.9), X.shape[0]-int(X.shape[0]*0.9)])

    seq2seq = Seq2Seq()
    trainer = pl.Trainer()
    trainer.fit(seq2seq, DataLoader(train), DataLoader(val))