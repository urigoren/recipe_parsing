from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
import numpy as np
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, T5Model
import pytorch_lightning as pl
from recipe_prep import *


class T5Recipe(T5ForConditionalGeneration):
    def set_generation_constraints(self, allowed_token_ids):
        """allowed_token_ids - list of lists, allowed token id by index"""
        self.constraints = torch.zeros(len(allowed_token_ids), self.config.vocab_size)
        for row, ids in enumerate(allowed_token_ids):
            for col in ids:
                self.constraints[row, col] = 1
        return self.constraints

    def generate(self, **kwargs):
        scores = super().generate(
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=1,
            **kwargs
        )['scores']
        ret = []
        for i, score in enumerate(scores):
            if i < self.constraints.shape[0]:
                ret.append((self.constraints[i] * F.softmax(score)).argmax().item())
            else:
                ret.append(score.argmax().item())
        return ret

class RecipeDataModule(pl.LightningDataModule):
    def __init__(self, csv_file,max_length = 64, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        output_vocab, id_types = t5_extra_vocab(31000)
        df = load_data_and_preprocess(csv_file, output_vocab, max_size=max_length)
        self.dataset_length = len(df)
        input_ids =         torch.Tensor(np.vstack(df["input_ids"])).type(torch.long)
        output_ids =        torch.Tensor(np.vstack(df["output_ids"])).type(torch.long)
        input_attention =   torch.Tensor(np.vstack(df["input_attention"])).type(torch.bool)
        output_attention =  torch.Tensor(np.vstack(df["output_attention"])).type(torch.bool)
        self.dataset = TensorDataset(input_ids, input_attention, output_ids, output_attention)

    def setup(self, stage=None):
        train_size = int(0.8 * self.dataset_length)
        valid_size = int(0.1 * self.dataset_length)
        test_size = self.dataset_length - valid_size - train_size
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset,
             [train_size, valid_size, test_size])
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class T5FineTune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, decoder_input_ids=None,decoder_attention_mask=None):
        # output = self.model(input_ids=input_ids, attention_mask=attention_mask,
        #                     decoder_input_ids=decoder_input_ids,
        #                     decoder_attention_mask=decoder_attention_mask)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=decoder_input_ids,
                            )
        return output.loss, output.logits
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
        loss, outputs = self.forward(input_ids,attention_mask,decoder_input_ids,decoder_attention_mask)
        self.log("Training loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
        loss, outputs = self.forward(input_ids,attention_mask,decoder_input_ids,decoder_attention_mask)
        self.log("validation_step loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
        loss, outputs = self.forward(input_ids,attention_mask,decoder_input_ids,decoder_attention_mask)
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        # labels = batch["labels"]
        self.log("Test loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    def save(self, p):
        Path(p).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)


def main(params):
    recipe_datamodule = RecipeDataModule(csv_file=params.data)
    tokenizer = T5Tokenizer.from_pretrained(params.token_model)
    model = T5Recipe.from_pretrained(params.load_model, return_dict=True)
    model.set_generation_constraints([[1], [1,2], [1], [1,2],
                                      # [1], [1,2], [1], [1,2],
                                      # [1], [1,2], [1], [1,2],
                                      [1], [1,2], [1], [1,2],])
    input_data = tokenizer("Studies have been shown that owning a dog is good for you",
                          return_tensors="pt").data
    # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").data
    # outputs = model(decoder_input_ids=decoder_input_ids, **input_data)
    outputs = model.generate(**input_data)
    print (outputs)


    trainer = pl.Trainer(max_epochs=10)
    t5finetune = T5FineTune(model)
    trainer.fit(t5finetune, recipe_datamodule)
    t5finetune.save(params.save_model)
    return 0


if __name__ == "__main__":
    import sys, os
    from argparse import ArgumentParser
    argparse = ArgumentParser()
    argparse.add_argument('--data', type=str, default='../data/seq2seq_4335716.csv')
    argparse.add_argument('--token_model', type=str, default='t5-large')
    argparse.add_argument('--load_model', type=str, default='t5-large')
    argparse.add_argument('--save_model', type=str, default='/home/ubuntu/trained_models/recipe_large_20')
    argparse.add_argument('--max_epochs', type=int, default=10)
    sys.exit(main(argparse.parse_args()))

