from dataclasses import dataclass
import os, json


@dataclass
class Config:
    def __init__(self, script_name):
        file_path = script_name[:-3]+'_config.json'
        assert os.path.isfile(file_path), f"error config file path: {file_path}"
        with open(file_path, 'r', encoding='utf-8') as fp:
            configs = json.load(fp)
        self.__dict__.update(**configs)
