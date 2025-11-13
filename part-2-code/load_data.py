import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
ENCODER_MAX_LENGTH = 256
DECODER_MAX_LENGTH = 256
BOS_TOKEN = "<extra_id_0>"
INPUT_PREFIX = "translate to sql: "
LOWERCASE_INPUTS = True

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    return _TOKENIZER

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = get_tokenizer()
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        if self.decoder_start_token_id is None:
            raise ValueError(f"Tokenizer missing BOS token {BOS_TOKEN}")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.has_targets = split != "test"
        self.samples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = self._resolve_path(data_folder, split, ["nl", "txt"])
        if nl_path is None:
            raise FileNotFoundError(f"Missing NL file for split '{split}' (looked for .nl/.txt)")
        with open(nl_path, "r") as f:
            nl_examples = [line.strip() for line in f.readlines()]

        if self.has_targets:
            sql_path = self._resolve_path(data_folder, split, ["sql", "txt"])
            if sql_path is None:
                raise FileNotFoundError(f"Missing SQL file for split '{split}' (looked for .sql/.txt)")
            with open(sql_path, "r") as f:
                sql_examples = [line.strip() for line in f.readlines()]
            if len(sql_examples) != len(nl_examples):
                raise ValueError(f"Mismatch between NL and SQL counts for split '{split}'")
        else:
            sql_examples = [""] * len(nl_examples)

        processed = []
        for nl_text, sql_text in zip(nl_examples, sql_examples):
            normalized_nl = nl_text.strip()
            if LOWERCASE_INPUTS:
                normalized_nl = normalized_nl.lower()
            if INPUT_PREFIX:
                normalized_nl = f"{INPUT_PREFIX}{normalized_nl}"
            encoder_ids = tokenizer(
                normalized_nl,
                truncation=True,
                max_length=ENCODER_MAX_LENGTH,
                add_special_tokens=True,
            )["input_ids"]

            if self.has_targets:
                sql_tokenized = tokenizer(
                    sql_text,
                    truncation=True,
                    max_length=DECODER_MAX_LENGTH - 1,
                    add_special_tokens=False,
                )["input_ids"]
                decoder_inputs = [self.decoder_start_token_id] + sql_tokenized
                decoder_targets = sql_tokenized + [self.eos_token_id]
            else:
                decoder_inputs = []
                decoder_targets = []

            processed.append(
                {
                    "encoder_ids": torch.tensor(encoder_ids, dtype=torch.long),
                    "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long)
                    if decoder_inputs
                    else None,
                    "decoder_targets": torch.tensor(decoder_targets, dtype=torch.long)
                    if decoder_targets
                    else None,
                }
            )
        return processed

    def _resolve_path(self, data_folder, split, extensions):
        for ext in extensions:
            candidate = os.path.join(data_folder, f"{split}.{ext}")
            if os.path.exists(candidate):
                return candidate
        return None
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.has_targets:
            return (
                sample["encoder_ids"],
                sample["decoder_inputs"],
                sample["decoder_targets"],
            )
        return (sample["encoder_ids"],)

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_tensors = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]

    encoder_padded = pad_sequence(encoder_tensors, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_padded != PAD_IDX).long()
    decoder_input_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_target_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

    batch_size = encoder_padded.size(0)
    bos_token_id = get_tokenizer().convert_tokens_to_ids(BOS_TOKEN)
    initial_decoder_inputs = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)

    return (
        encoder_padded,
        encoder_mask,
        decoder_input_padded,
        decoder_target_padded,
        initial_decoder_inputs,
    )

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_tensors = [item[0] for item in batch]
    encoder_padded = pad_sequence(encoder_tensors, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_padded != PAD_IDX).long()

    bos_token_id = get_tokenizer().convert_tokens_to_ids(BOS_TOKEN)
    initial_decoder_inputs = torch.full((encoder_padded.size(0), 1), bos_token_id, dtype=torch.long)
    return encoder_padded, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(_resolve_file_with_exts(data_folder, "train", ["nl", "txt"]))
    train_y = load_lines(_resolve_file_with_exts(data_folder, "train", ["sql", "txt"]))
    dev_x = load_lines(_resolve_file_with_exts(data_folder, "dev", ["nl", "txt"]))
    dev_y = load_lines(_resolve_file_with_exts(data_folder, "dev", ["sql", "txt"]))
    test_x = load_lines(_resolve_file_with_exts(data_folder, "test", ["nl", "txt"]))
    return train_x, train_y, dev_x, dev_y, test_x


def _resolve_file_with_exts(data_folder, name, extensions):
    for ext in extensions:
        candidate = os.path.join(data_folder, f"{name}.{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Missing file for {name}; looked for extensions {extensions}")

