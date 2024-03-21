from transformers import AutoTokenizer
from dataclasses import asdict
from ..arguments import TokenizerArguments



def load_tokenizer(
        tokenizer_path: str,
        tokenizer_args: TokenizerArguments
    ):

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        **asdict(tokenizer_args)
    )

    return tokenizer