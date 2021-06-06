import json
import numpy as np
import pandas as pd
from transformers import BertLMHeadModel, BertTokenizer

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
