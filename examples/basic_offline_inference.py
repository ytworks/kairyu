"""vLLM's basic offline inference example, with only the import rewritten.

On a machine without vLLM this runs on the mock backend; with vLLM installed
the same code drives a real model.
"""

from kairyu import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
