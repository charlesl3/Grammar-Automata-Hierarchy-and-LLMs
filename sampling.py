from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "What are penguins?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- Greedy ---
greedy = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False      
)
print("\n[GREEDY]\n", tokenizer.decode(greedy[0], skip_special_tokens=True))

# --- Temperature Sampling ---
temp = model.generate(
    **inputs,
    max_new_tokens=40,
    temperature=1.2,
    do_sample=True
)
print("\n[TEMP=1.2]\n", tokenizer.decode(temp[0], skip_special_tokens=True))

# --- Top-k Sampling ---
topk = model.generate(
    **inputs,
    max_new_tokens=40,
    temperature=0.8,
    top_k=5,
    do_sample=True
)
print("\n[TOP-K=5]\n", tokenizer.decode(topk[0], skip_special_tokens=True))
