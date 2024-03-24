from dataclasses import dataclass, field

@dataclass
class PTDatasetArguments:
    text_chunck_size:int = field(
        default=1024,
        metadata={
            "help": "the size of each chunck in dataset, \
                the size of each chunck can be padding or truncked by dataset collator, \
                but a very small / big chunck may destory disrupting semantic coherence."
        }
    )
    encoding:str = field(
        default="utf8",
        metadata={
            "help": "the encoding to open the txt files."
        }
    )
    chunck_label:str = field(
        default="train",
        metadata={
            "help": "the label for text chuncks in Dataset."
        }
    )
    num_procs:int = field(
        default=4,
        metadata={
            "help": "number of processes used in dataset preprocess. default to 4."
        }
    )
    block_size:int = field(
        default=1024,
        metadata={
            "help": "for pt, it means the length of a tokenized text block, \
                so it is the size of one time input for model. \
                it will be automatically overwrited to tokenizer.model_max_length. \
                just in case, we give a default size."
        }
    )
    overwrite_cache:bool = field(
        default=False,
        metadata={
            "help": "whether to use cache."
        }
    )