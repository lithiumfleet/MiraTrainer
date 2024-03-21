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