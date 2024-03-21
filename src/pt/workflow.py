# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py, follow LLaMA-Factory.
# And thank you, LLaMA-Factory!
from transformers import Trainer


from ..loader import *
from ..arguments import *


def run_pt(
        model_path: str,
        model_args: ModelArguments,
        tokenizer_args: TokenizerArguments,
    ):

    # prepare for training
    ## load model, tokenizer, dataset, dataset collator and optimizer
    tokenizer = load_tokenizer(model_path, tokenizer_args)
    model = load_model(model_path, model_args)
    dataset = load_dataset()
    optimizer = load_optimizer()

    ## initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    # training

    # evaluation

