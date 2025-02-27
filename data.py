import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
import datasets
from datasets import Dataset
import psutil
from model.utils import get_tokenizer

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

# def preprocess_gsm8k(data_line):
#     question = json.loads(data_line)['src'].strip()
#     target = json.loads(data_line)['trg'].strip()

#     rationales = json.loads(data_line)['rationales'].strip()
#     cot_sequences = [[question, rationales + ' #### ' + target]]

#     return cot_sequences

def preprocess_gsm8k(data_line, multipass=False, hidden_thought=False):
    question = json.loads(data_line)['src'].strip()
    target = json.loads(data_line)['trg'].strip()

    if hidden_thought and multipass:
        return [[question, ' #### ' + target]]



    if multipass:
        rationales = json.loads(data_line)['rationales'].strip().split(" ")
        target = '#### ' + target

        cot_sequences = []
        rationales = [''] + rationales + [target]

        for i in range(len(rationales)-1):
            cot_sequences.append(tuple([question + ' ' + ' '.join(rationales[0:i+1]), rationales[i+1]]))
    
    else:
        rationales = json.loads(data_line)['rationales'].strip()
        cot_sequences = [[question, rationales + ' #### ' + target]]
    
    return cot_sequences

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict(examples['src'], return_attention_mask=False)["input_ids"]
        input_id_y = vocab_dict(examples['trg'], return_attention_mask=False)["input_ids"]
        # input_id_x = vocab_dict.encode_token(examples['src'])
        # input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
        return result_dict


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print('### decoded_tokenized_datasets...x_example', vocab_dict.decode(tokenized_datasets['input_id_x'][0]))
    print('### decoded_tokenized_datasets...y_example', vocab_dict.decode(tokenized_datasets['input_id_y'][0]))

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = vocab_dict.eos_token_id
            src = group_lst['input_id_x'][i]
            trg = group_lst['input_id_y'][i]


            # len_z = len(src) + len(trg)
            # stat_path = './stat_train_data' + 'gsm8k' + '.jsonl'
            # stat = open(stat_path, 'a')
            # print(json.dumps({"source": src, "target": trg, "len_z": len_z}), file=stat)
            # stat.close()

            while len(src) + len(trg) > seq_len - 2:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            # Inject [SEP] between source question and the target answer
            lst.append(src + vocab_dict("[SEP]")["input_ids"] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        # print('### decoded_input_ids example', vocab_dict.decode(group_lst['input_ids'][0]))
        return group_lst
    
    # Merge the x and y into z and mask the x
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict("[PAD]")["input_ids"][0], max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets

class TextDataset(TorchDataset):
    def __init__(self, text_datasets):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        out_kwargs = {}

        out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
        out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

        return out_kwargs



def finetune_get_dataset(name, mode, multipass, hidden_thought, block_size=128, data_dir="datasets/gsm8k"):
    if name != "gsm8k":
        assert False, f"only gsm8k is supported for finetuning, now providing {name}."

    print('#'*30, '\nLoading dataset {} from {}...'.format(name, data_dir))

    sentence_lst = {'src':[], 'trg': []}

    if mode == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_dir}/train.jsonl'
    elif mode == 'validation':
        print('### Loading form the VALID set...')
        path = f'{data_dir}/valid.jsonl'
    elif mode == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_dir}/test.jsonl'

    # Maximum number of data samples to load. Additional samples will be ignored.
    MAX_DATA_LEN = 10000000
    with open(path, 'r') as f_reader:
        for row in f_reader:
            if name == 'gsm8k':
                if mode in {'train', 'validation', 'test'}:
                    cot_sentences = preprocess_gsm8k(row, multipass=multipass, hidden_thought=hidden_thought)
                else:
                    assert False, f"Invaild data mode {mode} for gsm8k detected."
    
            else:
                assert False, f"only gsm8k is supported for finetuning, now providing {name}."
    
            for cot_sentence in cot_sentences:
                if len(sentence_lst['src']) >= MAX_DATA_LEN:
                    break
                sentence_lst['src'].append(cot_sentence[0])
                sentence_lst['trg'].append(cot_sentence[1])

    print('### Data samples...\n', sentence_lst['src'][:10], sentence_lst['trg'][:10])

    _Simon = True
    if _Simon:
        with open('./raw_train_data' + 'gsm8k' + '.jsonl', 'w') as f:
            for i in range(len(sentence_lst['src'])):
                print(json.dumps({"source": sentence_lst['src'][i], "target": sentence_lst['trg'][i]}), file=f)
    
    tokenizer = get_tokenizer()
    train_dataset = TextDataset(helper_tokenize(sentence_lst, vocab_dict=tokenizer, seq_len=block_size))

    return train_dataset


    





def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer = get_tokenizer(digit=True)
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    _Simon_finetune = True
    if _Simon_finetune:
        train_set = finetune_get_dataset(config.data.train, "train", config.data.multipass, config.data.hidden_thought, block_size=config.training.block_size)
        valid_set = finetune_get_dataset(config.data.valid, "validation", config.data.multipass, config.data.hidden_thought, block_size=config.training.block_size)

    else:
        train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
        valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

