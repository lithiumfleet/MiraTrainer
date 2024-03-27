from vllm import LLM, SamplingParams
from timethis import timethis


# hypers
model_path = "/data/lixubin/models/Qwen/Qwen1.5-1.8B"
prompts = [ "The future of AI is", ] * 5000
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=model_path)

with timethis("total time:"):
    outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")