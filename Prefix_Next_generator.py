import random


###############################################################
# 1. generate_prefixes(grammar)
#    - generates full terminal sentences
#    - extracts terminal prefixes
###############################################################
def generate_prefixes(grammar, start="S", num_sentences=20):
    prefixes = []

    for _ in range(num_sentences):

        # ---- generate ONE full terminal sentence ----
        current = [start]
        for _ in range(20):
            # fully terminal?
            if all(tok not in grammar for tok in current):
                break

            # expand first nonterminal
            for i, tok in enumerate(current):
                if tok in grammar:
                    expansions = []
                    for rhs in grammar[tok]:
                        if isinstance(rhs, str):     # terminal rule
                            expansions.append(current[:i] + [rhs] + current[i+1:])
                        else:                        # nonterminal rule
                            expansions.append(current[:i] + rhs + current[i+1:])
                    current = random.choice(expansions)
                    break

        # ---- extract terminal prefixes from this sentence ----
        # include empty prefix []
        prefixes.append([])

        for i in range(1, len(current)+1):
            prefixes.append(current[:i])

    return prefixes



###############################################################
# 2. get_allowed_next(prefix, grammar)
#    - compute legal next terminals after consuming "prefix"
###############################################################
def get_allowed_next(prefix, grammar):
    allowed = set()
    queue = [["S"]]

    while queue:
        cur = queue.pop()

        # try to consume prefix from cur
        i = 0
        while i < len(prefix) and i < len(cur) and cur[i] not in grammar:
            if cur[i] != prefix[i]:
                break
            i += 1

        if i == len(prefix):
            # we matched the whole prefix
            rest = cur[i:]

            # next is a terminal
            if rest and rest[0] not in grammar:
                allowed.add(rest[0])
                continue

            # next is a nonterminal → expand it
            if rest:
                nt = rest[0]
                for rhs in grammar[nt]:
                    if isinstance(rhs, str):
                        new = cur[:i] + [rhs] + rest[1:]
                    else:
                        new = cur[:i] + rhs + rest[1:]
                    queue.append(new)
            continue

        # mismatch: expand first nonterminal in cur
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



###############################################################
# Example Usage
###############################################################
if __name__ == "__main__":

    grammar = {
        "Det": ["glar", "flem"],
        "Adj": ["wug", "zarp"],
        "N":   ["blick", "snorp", "flarm"],
        "V":   ["drimps", "glings", "flops"],
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

    prefixes = generate_prefixes(grammar, num_sentences=5)

    for p in prefixes:
        print("prefix =", p, " -> allowed =", get_allowed_next(p, grammar))
