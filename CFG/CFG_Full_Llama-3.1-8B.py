import random
import requests


##########################################################################
# GRAMMAR TEACHING PROMPT
##########################################################################

grammar_description = """
You are a next-token generator for a context-free grammar.

Here is the grammar:

Det → glar | flem | okurrr
Adj → wug | zarp | cpdd
N   → blick | snorp | flarm | fcccc
V   → drimps | glings | flops | cooo
NP  → Det N | Det Adj N 
VP  → V | V NP
S   → NP VP

Your job:
Given a prefix consisting ONLY of terminal words,
predict the next TERMINAL TOKEN that is legal under the grammar.

Rules:
- Output ONLY the next terminal token.
- Do NOT add punctuation.
- Do NOT add explanations.
- Do NOT complete the whole sentence.
- Output exactly ONE token.
"""



##########################################################################
# 1. Generate terminal prefixes
##########################################################################

def generate_prefixes(grammar, start="S", num_sentences=20):
    prefixes = []

    for _ in range(num_sentences):

        current = [start]

        # generate one terminal sentence
        for _ in range(20):
            if all(tok not in grammar for tok in current):
                break

            # expand first nonterminal
            for i, tok in enumerate(current):
                if tok in grammar:
                    expansions = []
                    for rhs in grammar[tok]:
                        if isinstance(rhs, str):
                            expansions.append(current[:i] + [rhs] + current[i+1:])
                        else:
                            expansions.append(current[:i] + rhs + current[i+1:])
                    current = random.choice(expansions)
                    break

        # extract prefixes
        prefixes.append([])  # empty prefix
        for i in range(1, len(current)+1):
            prefixes.append(current[:i])

    return prefixes



##########################################################################
# 2. Ground-truth allowed-next terminals
##########################################################################

def get_allowed_next(prefix, grammar):
    allowed = set()
    queue = [["S"]]

    while queue:
        cur = queue.pop()

        # match prefix
        i = 0
        while i < len(prefix) and i < len(cur) and cur[i] not in grammar:
            if cur[i] != prefix[i]:
                break
            i += 1

        if i == len(prefix):
            rest = cur[i:]

            if rest and rest[0] not in grammar:
                allowed.add(rest[0])
                continue

            if rest:
                nt = rest[0]
                for rhs in grammar[nt]:
                    if isinstance(rhs, str):
                        new = cur[:i] + [rhs] + rest[1:]
                    else:
                        new = cur[:i] + rhs + rest[1:]
                    queue.append(new)
            continue

        # mismatch → expand first nonterminal
        for j, tok in enumerate(cur):
            if tok in grammar:
                for rhs in grammar[tok]:
                    if isinstance(rhs, str):
                        new = cur[:j] + [rhs] + cur[j+1:]
                    else:
                        new = cur[:j] + rhs + cur[j+1:]
                    queue.append(new)
                break

    return allowed



##########################################################################
# 3. Predict next token using LOCAL Llama 3.1 8B (Ollama)
##########################################################################

def predict_next(prefix):
    prefix_str = " ".join(prefix)
    if prefix_str != "":
        prefix_str += " "

    prompt = grammar_description + "\n\nPrefix: " + prefix_str + "\nNext terminal: "

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "max_tokens": 10
        }
    )

    raw = response.json()["response"].strip()
    if raw == "":
        return ""

    return raw.split()[0]    # take only the first token



##########################################################################
# 4. Evaluation (prefix → allowed → model prediction → correctness)
##########################################################################

def evaluate(prefixes, grammar):
    results = []

    for prefix in prefixes:
        allowed = get_allowed_next(prefix, grammar)
        pred = predict_next(prefix)
        correct = pred in allowed

        results.append({
            "prefix": prefix,
            "allowed": list(allowed),
            "pred": pred,
            "correct": correct
        })

    return results



##########################################################################
# 5. Accuracy
##########################################################################

def compute_accuracy(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall_acc = correct / total if total > 0 else 0

    # accuracy by prefix length
    length_stats = {}
    for r in results:
        L = len(r["prefix"])
        if L not in length_stats:
            length_stats[L] = {"correct": 0, "total": 0}
        length_stats[L]["total"] += 1
        if r["correct"]:
            length_stats[L]["correct"] += 1

    length_acc = {}
    for L, d in length_stats.items():
        length_acc[L] = d["correct"] / d["total"]

    return overall_acc, length_acc



##########################################################################
# 6. Runner (Full Pipeline)
##########################################################################

def run_llama(grammar, num_sentences=10):
    prefixes = generate_prefixes(grammar, num_sentences=num_sentences)
    results = evaluate(prefixes, grammar)
    overall_acc, length_acc = compute_accuracy(results)

    print("\n=== ACCURACY SUMMARY ===")
    print("Total predictions:", len(results))
    print("Overall accuracy:", round(overall_acc, 4))

    print("\nAccuracy by prefix length:")
    for L in sorted(length_acc.keys()):
        print("  Length", L, ":", round(length_acc[L], 4))

    print("=========================\n")

    return results



##########################################################################
# Example Usage
##########################################################################

if __name__ == "__main__":

    grammar = {
        "Det": ["glar", "flem", "okurrr"],

        "Adj": ["wug", "zarp", "cpdd"],

        "N": ["blick", "snorp", "flarm", "fcccc"],

        "V": ["drimps", "glings", "flops", "cooo"],

        "NP": [
            ["Det", "N"],
            ["Det", "Adj", "N"]
        ],

        "VP": [
            ["V"],
            ["V", "NP"]
        ],

        "S": [
            ["NP", "VP"]
        ]
    }

    print("Running Llama 3.1 8B evaluation...\n")
    results = run_llama(grammar, num_sentences=5)

    for r in results:
        print(r)
"""
=== ACCURACY SUMMARY ===
Total predictions: 29
Overall accuracy: 0.2759

Accuracy by prefix length:
  Length 0 : 0.4
  Length 1 : 0.6
  Length 2 : 0.2
  Length 3 : 0.2
  Length 4 : 0.0
  Length 5 : 0.5
  Length 6 : 0.0
  Length 7 : 0.0
=========================
"""
