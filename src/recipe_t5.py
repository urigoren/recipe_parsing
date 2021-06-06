import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, T5Model
from recipe_prep import *
if __name__ == "__main__":
    max_length = 64
    output_vocab, id_types = unified_vocab()
    df = load_data_and_preprocess('../data/seq2seq_4335716.csv', output_vocab, max_size=max_length)
    input_ids =         torch.Tensor(np.vstack(df["input_ids"])).type(torch.int)
    output_ids =        torch.Tensor(np.vstack(df["output_ids"])).type(torch.int)
    input_attention =   torch.Tensor(np.vstack(df["input_attention"])).type(torch.bool)
    output_attention =  torch.Tensor(np.vstack(df["output_attention"])).type(torch.bool)

    class T5Recipe(T5ForConditionalGeneration):
        def set_generation_constraints(self, allowed_token_ids):
            """allowed_token_ids - list of lists, allowed token id by index"""
            self.constraints = torch.zeros(len(allowed_token_ids), self.config.vocab_size)
            for row, ids in enumerate(allowed_token_ids):
                for col in ids:
                    self.constraints[row, col]=1
            return self.constraints
        def generate(self, **kwargs):
            print ("BOOM")
            scores = super().generate(
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=1,
                **kwargs
            )['scores']
            ret = []
            for i, score in enumerate(scores):
                if i<self.constraints.shape[0]:
                    ret.append((self.constraints[i]*F.softmax(score)).argmax().item())
                else:
                    ret.append(score.argmax().item())
            return ret

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Recipe.from_pretrained('t5-small')
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
