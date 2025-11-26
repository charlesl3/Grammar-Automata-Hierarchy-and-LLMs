from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",     # MPS if available
    torch_dtype=torch.float16,
)
print("Loaded on:", model.hf_device_map)

prompt = "User: what is the capital of United States?\nAssistant:"
#prompt = "what is the capital of United States?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,       # random sampling for creativity
    top_p=0.9,
    temperature=0.8
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
