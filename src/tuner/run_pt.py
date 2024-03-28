from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, default_data_collator, TrainingArguments
from dataclasses import dataclass
from data_utils import get_dataset
import json
import os

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Config:
    def __init__(self):
        file_path = __file__[:-3]+'_config.json'
        assert os.path.isfile(file_path), f"error config file path: {file_path}"
        with open(file_path, 'r', encoding='utf-8') as fp:
            configs = json.load(fp)
        self.__dict__.update(**configs)

class Task:
    def __init__(self) -> None:
        self.cfg = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.dataset = get_dataset(self.tokenizer, self.cfg.dataset_director, 1024)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        self.trining_args = TrainingArguments(
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
            args=self.trining_args
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
