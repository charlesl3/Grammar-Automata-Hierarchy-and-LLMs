from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"    # base GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = """Using ONLY the grammar below, generate ONE valid sentence.
Do not explain. Output only the sentence.

Grammar:
S  → NP VP
NP → Det Adj N | Det N
VP → V | V NP
Det → glar | flem
Adj → wug | zarp | fck
N   → blick | snorp | flarm
V   → drimps | glings | flops
"""

prompt2 = """
You are given the following grammar rules:

S  → NP VP
NP → Det Adj N | Det N
VP → V | V NP
Det → glar | flem
Adj → wug | zarp
N   → blick | snorp | flarm
V   → drimps | glings | flops

Task:
Given ONLY these rules, determine the next valid token
that can legally follow the prefix below.

Respond with ONLY the next token, nothing else.

Prefix: glar blick
what is the next token?
"""

inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    temperature=0.2
)

print("\nGPT-2 Output:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))



model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)

# output = model.generate(
#     **inputs,
#     max_new_tokens=40,
#     temperature=0.8,
#     top_p=0.9,
#     do_sample=True
# )

output = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False,
    temperature=0.2
)


print("\nPhi-3 Output:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
