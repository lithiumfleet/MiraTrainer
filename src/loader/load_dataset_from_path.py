from datasets import Dataset
from typing import Optional, Generator
from os.path import isfile
from dataclasses import asdict
from ..arguments import PTDatasetArguments





def load_multiple_files(dataset_path:list[str], encoding:str, chunck_label:str) -> Generator:
    all_text_lines = list()
    for file_path in dataset_path:
        if not isfile(file_path):
            raise FileNotFoundError(f"{file_path} can't touch! Please check your 'dataset_path' again.")

        with open(file_path, 'r', encoding=encoding) as fp:
            for line in fp.readlines():
                all_text_lines.append({chunck_label: line})
    
    return all_text_lines








def load_dataset_from_path(
        dataset_path: Optional[str|list[str]],
        dataset_args: PTDatasetArguments,
        ) -> Dataset:

    r"""
        this function only support load from txt file paths which specified in dataset_path.
    """

    # when dataset_path is only a str, convert it to list[str]
    if isinstance(dataset_path, str):
        dataset_path = [dataset_path]

    # thanks! https://github.com/huggingface/datasets/issues/6118
    dataset = Dataset.from_list(load_multiple_files(dataset_path,dataset_args.encoding,dataset_args.chunck_label))

    return dataset