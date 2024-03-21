from src import ModelArguments, TokenizerArguments, run_pt

# for debug usage
if __name__ == '__main__':
    model_path = '/data/lixubin/models/gpt2'
    model_args = ModelArguments()
    tokenizer_args = TokenizerArguments()
    run_pt(model_path,model_args,tokenizer_args)