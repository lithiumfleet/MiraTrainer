from src import *

# for debug usage
if __name__ == '__main__':
    model_path = '/data/lixubin/models/gpt2'
    dataset_path = '/data/lixubin/MiraTrainer/test.txt'
    model_args = ModelArguments()
    tokenizer_args = TokenizerArguments()
    dataset_args = PTDatasetArguments()
    run_pt(model_path,model_args,tokenizer_args,dataset_path,dataset_args)