from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator
from peft import LoraConfig
from trl import SFTTrainer
from dataclasses import dataclass
import os, json
from sft_data_utils import get_sft_dataset

# close warning: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.dataset = get_sft_dataset(self.cfg.dataset_director, convert_sharegpt_to_vicuna=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        self.training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=10,
            learning_rate=4e-5,
            seed=42,
            logging_steps=1,
            output_dir=self.cfg.output_dir,
            dataloader_num_workers=40
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
            args=self.training_args,
            max_seq_length=2048,
            # peft_config=self.lora_config ######## NOTICING: Setting LoRA Here!! ##########
        )
        # training
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = self.dataset.num_rows['train'] # set 'train' colum or here will be a crash. num_rows = {'train': number}

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == '__main__':
    task = SFTTask()
    task.run_sft()