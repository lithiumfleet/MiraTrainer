from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig
from script_config import Config
from dpo_data_utils import get_dpo_dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DPOTask:
    def __init__(self) -> None:
        self.cfg = Config(__file__)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_path)
        self.dataset = get_dpo_dataset(self.cfg.dataset_director)
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

    def run_dpo(self):
        # Fix Tokenizer bug in DPOTrainer For Qwen: https://github.com/huggingface/trl/issues/1073
        self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.eos_token})
        self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        ## initialize trainer
        trainer = DPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            args=self.training_args,
            max_length=1536,
            max_prompt_length=1024,
            max_target_length=512,
            peft_config=self.lora_config ######## NOTICING: Setting LoRA Here!! ##########
        )
        # training
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = self.dataset.num_rows

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == '__main__':
    task = DPOTask()
    task.run_dpo()