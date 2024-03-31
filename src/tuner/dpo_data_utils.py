from datasets import load_dataset


def get_dpo_dataset(dataset_dir:str):
    dataset = load_dataset(path=dataset_dir, split='train')
    return dataset