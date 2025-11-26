import random
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype="auto",
                                             device_map="auto")

# prefixes to test
prefixes = [
    # Det Adj N (12)
    "goayy wug blick",
    "goayy wug snorp",
    "goayy wug flarm",
    "goayy zarp blick",
    "goayy zarp snorp",
    "goayy zarp flarm",
    "flem wug blick",
    "flem wug snorp",
    "flem wug flarm",
    "flem zarp blick",
    "flem zarp snorp",
    "flem zarp flarm",

    # Det N (8)
    "goayy blick",
    "goayy snorp",
    "goayy flarm",
    "flem blick",
    "flem snorp",
    "flem flarm",

    # Mixed examples
    "goayy wug flarm",
    "flem zarp blick"
]

allowed = {"drimps", "glings", "flops"}


def test_prefix(prefix):
    prompt = f"""
You are given these grammar rules:

S → NP VP
NP → Det Adj N | Det N
VP → V | V NP
Det → goayy | flem
Adj → wug | zarp
N   → blick | snorp | flarm
V   → drimps | glings | flops

Task:
Return only ONE token: the next valid token after this prefix; exactly one word, no others.

Prefix: {prefix}
Next token:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=70,
        do_sample=False,
        temperature=0.0
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split()[-1]  # last token
    return answer

#print(test_prefix(prefixes[1]))
# evaluation
correct = 0
total = len(prefixes)

for p in prefixes:
    pred = test_prefix(p)
    print(f"Prefix: {p} | Model predicted: {pred}")
    if pred in allowed:
        correct += 1

acc = correct / total
print("\nAccuracy:", acc) #5%


# GPT-2 performs extremely poorly on CFG next-token tests because it lacks instruction tuning
# and cannot interpret or apply grammar rules. It does not constrain its output to the allowed
# terminal set and instead produces arbitrary English-like continuations. Since GPT-2 relies
# purely on surface-level statistical patterns rather than symbolic rule following, its accuracy
# is effectively zero on this task.
