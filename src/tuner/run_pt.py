from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, default_data_collator, TrainingArguments
from data_utils import get_dataset
import os
from script_config import Config

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Task:
    def __init__(self) -> None:
        self.cfg = Config(__file__)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.dataset = get_dataset(self.tokenizer, self.cfg.dataset_director, 1024)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        self.training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            seed=42,
            logging_steps=1,
            output_dir=self.cfg.output_dir,
            dataloader_num_workers=24
        )

    # main function
    def run_pt(self):
        ## initialize trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            data_collator=default_data_collator,
            args=self.training_args
        )

        # training
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = self.dataset.num_rows

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
            

if __name__ == '__main__':
    task = Task()
    task.run_pt()
