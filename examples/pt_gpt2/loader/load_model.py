from transformers import AutoModelForCausalLM
from dataclasses import asdict
from ..arguments import ModelArguments

def load_model(
        model_path: str,
        model_args: ModelArguments
    ):

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        **asdict(model_args)
    )

    return model
