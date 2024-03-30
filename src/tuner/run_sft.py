from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator
from peft import LoraConfig
from trl import SFTTrainer
from dataclasses import dataclass
import os, json
from sft_data_utils import get_sft_dataset


@dataclass
class Config:
    def __init__(self):
        file_path = __file__[:-3]+'_config.json'
        assert os.path.isfile(file_path), f"error config file path: {file_path}"
        with open(file_path, 'r', encoding='utf-8') as fp:
            configs = json.load(fp)
        self.__dict__.update(**configs)


class SFTTask:
    def __init__(self) -> None:
        self.cfg = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.dataset = get_sft_dataset(self.cfg.dataset_director, self.tokenizer, True) #FIXME: BUG here, need more info about SFTTrainer traindatset format info.
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
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","v_proj"]
        )

    def run_sft(self):
        ## initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            data_collator=default_data_collator,
            args=self.training_args,
            peft_config=self.lora_config,
            max_seq_length=2048
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
    task = SFTTask()
    task.run_sft()