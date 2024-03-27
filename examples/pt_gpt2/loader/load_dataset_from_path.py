from datasets import Dataset
from typing import Optional, Generator
from os.path import isfile
from dataclasses import asdict
from ..arguments import PTDatasetArguments
from transformers import PreTrainedTokenizer
from functools import partial
from itertools import chain


def load_multiple_files(dataset_path:list[str], encoding:str, chunck_label:str) -> list[str]:
    # when dataset_path is only a str, convert it to list[str]
    if isinstance(dataset_path, str):
        dataset_path = [dataset_path]

    all_text_lines = list()
    for file_path in dataset_path:
        if not isfile(file_path):
            raise FileNotFoundError(f"{file_path} can't touch! Please check your 'dataset_path' again.")

        with open(file_path, 'r', encoding=encoding) as fp:
            for line in fp.readlines():
                all_text_lines.append({chunck_label: line})
    
    return all_text_lines



def group_texts(examples, block_size:int):
    # example structure: [ { input_ids:[...], attention_mask:[...] } * n

    # concat all tokens of inputids / masks
    concatenated_examples:dict = {
        'input_ids': list(chain(*examples['input_ids'])),
        'attention_mask': list(chain(*examples['attention_mask'])),
    }
    assert len(concatenated_examples['attention_mask']) == len(concatenated_examples['input_ids']), "Unequal length between input_ids and attention_mask in dataset."

    total_length = len(concatenated_examples['input_ids'])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    total_length = (total_length // block_size) * block_size

    # Split by chunk_size.
    result = {
        'input_ids': [
            concatenated_examples['input_ids'][i : i + block_size] 
            for i in range(0, total_length, block_size)
        ],
        'attention_mask': [
            concatenated_examples['attention_mask'][i : i + block_size] 
            for i in range(0, total_length, block_size)
        ]
    }
    result['labels'] = result['input_ids'].copy()
    return result




def load_dataset_from_path(
        dataset_path: Optional[str|list[str]],
        tokenizer: PreTrainedTokenizer,
        dataset_args: PTDatasetArguments,
    ) -> Dataset:
    r"""
        this function only support load from txt file paths which specified in dataset_path.
    """
    # thanks! https://github.com/huggingface/datasets/issues/6118. (but no generators here.)
    raw_dataset = Dataset.from_list(load_multiple_files(dataset_path,dataset_args.encoding,dataset_args.chunck_label))

    tokenized_datset = raw_dataset.map(
        lambda example: tokenizer(example[dataset_args.chunck_label]),
        batched = True,
        num_proc = dataset_args.num_procs,
        remove_columns=dataset_args.chunck_label,
        desc = "Running tokenizer on dataset..."
    )

    # set block_size and pass it as default argument to group_texts
    # expect internml: block_size = tokenizer.model_max_length if tokenizer.model_max_length != None else dataset_args.block_size
    # FIXME: internml max_model_length toooo big
    block_size =  dataset_args.block_size
    group_texts_with_block_size = partial(group_texts, block_size=block_size)
    
    lm_dataset = tokenized_datset.map(
        group_texts_with_block_size,
        batched=True,
        num_proc=dataset_args.num_procs,
        load_from_cache_file=not dataset_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_dataset