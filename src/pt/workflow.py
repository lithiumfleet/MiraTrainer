# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py, follow LLaMA-Factory.
# And thank you, LLaMA-Factory!
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from typing import Optional
import os

from ..loader import *
from ..arguments import *
 


def get_checkpoint_or_resume_from_last(training_args: TrainingArguments):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def run_pt(
        model_path: str,
        model_args: ModelArguments,
        tokenizer_args: TokenizerArguments,
        dataset_path: Optional[str|list[str]],
        dataset_args: PTDatasetArguments,
        training_args: TrainingArguments
    ):
    # prepare for training
    ## load model, tokenizer, dataset, dataset collator and optimizer
    tokenizer = load_tokenizer(model_path, tokenizer_args)
    model = load_model(model_path, model_args)
    dataset = load_dataset_from_path(dataset_path, tokenizer, dataset_args)

    ## initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=default_data_collator,
        args=training_args
    )

    ## Detecting last checkpoint.
    checkpoint = get_checkpoint_or_resume_from_last(training_args)

    # training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = dataset.num_rows

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
