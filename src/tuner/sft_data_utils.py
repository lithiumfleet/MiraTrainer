from datasets import load_dataset

def trans_sharegpt_to_vicuna(batch):
    """
    an example in Vicuna(?) format: 
        {
            "id": 1,
            "messages": [
                { "role": "user", "content": "" },
                { "role": "assistant", "content": "" },
                { "role": "user", "content": "" },
                { "role": "assistant", "content": "" }
            ]
        }
    """
    # FIXME: Remember to do code review.
    messages = []
    for index in range(len(batch['instruction'])):
        history:list[list[str]] = batch['history'][index] + [[batch['instruction'][index]+batch['input'][index], batch['output'][index]]]
        assert len(history[0]) == len(history[-1]) == 2

        conversation:list[dict[str:str]] = []
        for pair in history:
            conversation += [{"role":"user", "content":pair[0]}, {"role":"assistant", "content":pair[1]}]
        assert len(conversation) >= 2 and len(conversation)%2 == 0

        for i in range(2, len(conversation)+1, 2):
            messages.append(conversation[:i])
    return {"messages": messages}



def get_sft_dataset(dataset_path: str, convert_sharegpt_to_vicuna:bool=False):
    """
    only support vicuna format.
    """
    dataset = load_dataset(dataset_path)
    if convert_sharegpt_to_vicuna:
        dataset = dataset.map(
            trans_sharegpt_to_vicuna,
            load_from_cache_file=False, # no cache....
            remove_columns=['instruction', 'input', 'history', 'output'],
            batched=True,
            num_proc=40
        )

    return dataset

