from transformers import TrainingArguments
from src import *

# for debug usage
if __name__ == '__main__':
    model_path = '/data/lixubin/models/gpt2'
    dataset_path = '/data/lixubin/MiraTrainer/test.txt'
    model_args = ModelArguments()
    tokenizer_args = TokenizerArguments()
    dataset_args = PTDatasetArguments()
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        do_train=True,
        output_dir="/tmp/test-clm",
        overwrite_output_dir=True
    )
    run_pt(model_path,model_args,tokenizer_args,dataset_path,dataset_args,training_args)