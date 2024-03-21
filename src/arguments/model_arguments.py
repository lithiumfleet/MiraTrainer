from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    trust_remote_code:str = field(
        default = True,
        metadata = {
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should only be set to True for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine."
        }
    )
    