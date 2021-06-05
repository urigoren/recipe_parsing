import torch
from datasets import load_dataset, load_metric, list_datasets, list_metrics
from transformers import BertTokenizer, EncoderDecoderModel, BasicTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


import json
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Split

output_vocab = {
    "[PAD]": 0,
    "[BOS]": 1,
    "PUT":2,
    "REMOVE":3,
    "USE":4,
    "STOP_USING":5,
    "CHEF_CHECK":6,
    "CHEF_DO":7,
    "MOVE_CONTENTS":8,
}
k = len(output_vocab)
with open("../data/res2idx.json", 'r') as f:
    for w,i in json.load(f).items():
        output_vocab[w] = k
        k+=1
with open("../data/arg2idx.json", 'r') as f:
    for w,i in json.load(f).items():
        output_vocab[w.replace('-','_')] = k
        k+=1

output_vocab = {w:i for i,w in enumerate(output_vocab)}
output_tokenizer = Tokenizer(WordLevel(output_vocab,))
output_tokenizer.pre_tokenizer = Whitespace()

t = output_tokenizer.encode_batch(["SERVE MOVE_CONTENTS","SERVE MOVE_CONTENTS PUT"])
# print (t)

csv_file = '../data/seq2seq_4335716.csv'
input_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_tokenizer.bos_token = input_tokenizer.cls_token
input_tokenizer.eos_token = input_tokenizer.sep_token

val_data = load_dataset('csv', data_files=csv_file, split='train[90%:]')
train_data = load_dataset('csv', data_files=csv_file, split='train[:90%]')
# print(val_data)
# print(train_data)

batch_size = 16  # 4 but change to 16 for full training
encoder_max_length = 128
decoder_max_length = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    tok_params = {
        'padding': 'max_length',
        'truncation': True,
        'max_length': encoder_max_length,
    }
    inputs = input_tokenizer(batch['input_seq'], **tok_params)
    # outputs = output_tokenizer(batch['output_seq'], **tok_params)
    outputs = output_tokenizer.encode_batch(batch['output_seq'])
    # pad
    padded_output = np.zeros((len(outputs), decoder_max_length - 1))
    attention_output = np.zeros((len(outputs), decoder_max_length))
    for i, arr in enumerate(outputs):
        padded_output[i, :len(arr.ids)] = arr.ids
        attention_output[i, :len(arr.ids)] = 1
    padded_output = np.concatenate([np.ones((len(outputs), 1)), padded_output], axis=1).astype(int)
    attention_output = attention_output.astype(int)

    batch['input_ids'] = inputs.input_ids
    batch['attention_mask'] = inputs.attention_mask
    batch['decoder_input_ids'] = padded_output
    batch['decoder_attention_mask'] = attention_output
    batch['labels'] = padded_output

# only use 32 training examples for notebook - COMMENT LINE FOR FULL TRAINING
#train_data = train_data.select(range(32))

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
#     remove_columns=['name', 'note'],
)
train_data.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'],
)

# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
# val_data = val_data.select(range(16))

val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
#     remove_columns=['name', 'note'],
)
val_data.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'],
)

ed_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

# set special tokens
ed_model.config.decoder_start_token_id = 1
ed_model.config.eos_token_id = input_tokenizer.eos_token_id
ed_model.config.pad_token_id = input_tokenizer.pad_token_id

# sensible parameters for beam search
ed_model.config.vocab_size = len(output_vocab)
ed_model.config.max_length = 142
ed_model.config.min_length = 56
ed_model.config.no_repeat_ngram_size = 3
ed_model.config.early_stopping = True
ed_model.config.length_penalty = 2.0
ed_model.config.num_beams = 4


# load wer for validation
wer = load_metric('wer')


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = output_tokenizer.decode_batch(pred_ids)
    #labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = output_tokenizer.decode_batch(labels_ids)
    #label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    wer_output = wer.compute(predictions=pred_str, references=label_str)

    return {
        'wer': round(wer_output, 4),
    }


# set training arguments - these params are not really tuned, feel free to change
# training_args = Seq2SeqTrainingArguments(
#     output_dir='./',
#     evaluation_strategy='steps',
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_with_generate=True,
#     logging_steps=500,  # 2 or set to 1000 for full training
#     save_steps=500,  # 16 or set to 500 for full training
#     eval_steps=500,  # 4 or set to 8000 for full training
#     warmup_steps=500,  # 1 or set to 2000 for full training
#     max_steps=2500,  # 16 or comment for full training
#     overwrite_output_dir=True,
#     save_total_limit=3,
#     fp16=torch.cuda.is_available(),
# )

training_args = Seq2SeqTrainingArguments(
    output_dir='./',
    evaluation_strategy='steps',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=50,  # 2 or set to 1000 for full training
    save_steps=50,  # 16 or set to 500 for full training
    eval_steps=50,  # 4 or set to 8000 for full training
    warmup_steps=50,  # 1 or set to 2000 for full training
    max_steps=850,  # 16 or comment for full training
    overwrite_output_dir=True,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=ed_model,
    tokenizer=input_tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()
