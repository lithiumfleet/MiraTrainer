from transformers import PreTrainedTokenizer
from datasets import load_dataset
from functools import partial

def trans_sharegpt_to_vicuna(example, tokenizer:PreTrainedTokenizer):
    """
    an example in Vicuna format: 
        {
            "id": 1,
            "conversations": [
                { "role": "user", "content": "" },
                { "role": "assistant", "content": "" },
                { "role": "user", "content": "" },
                { "role": "assistant", "content": "" }
            ]
        }
    """
    history = example['history']+[example['instruction']+example['input'], example['output']]
    converstaions = []
    for pair in history:
        converstaions.append( {"role":"user",      "content":pair[0]} )
        converstaions.append( {"role":"assistant", "content":pair[1]} )

    return { "conversations": converstaions }

def get_sft_dataset(dataset_path: str, tokenizer:PreTrainedTokenizer, convert_sharegpt_to_vicuna:bool=False):
    dataset = load_dataset(dataset_path)
    collate_func = partial(trans_sharegpt_to_vicuna, tokenizer=tokenizer)
    if convert_sharegpt_to_vicuna:
        dataset = dataset.map(
            collate_func,
            remove_columns=['instruction', 'input', 'history', 'output']
        )
    return dataset

