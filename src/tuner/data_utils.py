from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import os
from functools import partial
from torch import ones
from tqdm import tqdm
import re
import torch


def clean(text:str):
    url_pattern = r"[(轻小说文库)(Www.WenKu8.Com)(hayamimashiro)]"
    special_symbols = r"[`@#$%^&*()_+-=<>/'\"\\\[\]{}◇◆『』(※)：～「」─…★☆]"
    whitespace_pattern = r"\s+"
    text = re.sub(url_pattern, "", text)
    text = re.sub(special_symbols, "", text)
    text = re.sub(whitespace_pattern, " ", text)
    return text
    



def clean_and_group_one_doc(doc:str, tokenizer:PreTrainedTokenizer, chuncksize:int):
    with open(doc, 'r') as doc_fp:
        text = doc_fp.read()
    text = clean(text)
    result = tokenizer(text, return_tensors='pt')
    input_ids = result['input_ids'][0]
    attention_mask = result['attention_mask'][0]
    total_length = (len(input_ids) // chuncksize) * chuncksize
    return [
        {
            "input_ids":input_ids[i:i+chuncksize],
            "attention_mask":attention_mask[i:i+chuncksize],
            "labels":input_ids[i:i+chuncksize]
        }
        for i in range(0, total_length, chuncksize)
    ]



def get_dataset(tokenizer:PreTrainedTokenizer, dataset_director:str, chuncksize:int) -> Dataset:
    docs = [dataset_director+os.sep+doc for doc in os.listdir(dataset_director)]
    doc_handler = partial(clean_and_group_one_doc, tokenizer=tokenizer, chuncksize=chuncksize)
    dataset = []
    for doc in tqdm(docs):    
        dataset += doc_handler(doc)
    lm_dataset = Dataset.from_list(dataset)
    return lm_dataset

