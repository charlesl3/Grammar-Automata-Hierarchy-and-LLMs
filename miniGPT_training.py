import math
import torch
import torch.nn as nn
import torch.nn.functional as F




device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size   = 1000    # toy vocabulary
d_model      = 32      # hidden size
num_heads    = 4       # must divide d_model
d_head       = d_model // num_heads  # = 8
ffn_hidden   = 64      # hidden size in FFN
max_seq_len  = 32      # context length
n_layers     = 2       # number of decoder blocks

batch_size   = 16
num_steps    = 200      # training iterations
lr           = 1e-3     # learning rate




def causal_mask(seq_len):
    """
    Create a [1, 1, L, L] causal mask where positions
    cannot attend to future tokens.

    mask[i, j] = -inf when j > i, else 0.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask.view(1, 1, seq_len, seq_len)  # [1,1,L,L]




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        """
        x:    [B, L, d_model]
        mask: [1, 1, L, L]
        """
        B, L, D = x.shape  


        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  

        # Add causal mask
        scores = scores + mask  


        attn = F.softmax(scores, dim=-1)  


        out = attn @ v


        out = out.transpose(1, 2).contiguous().view(B, L, D)


        out = self.W_o(out)  
        return out



class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        x = self.fc1(x)      # [B, L, hidden_dim]
        x = F.gelu(x)
        x = self.fc2(x)      # [B, L, d_model]
        return x




class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden)

    def forward(self, x, mask):
        """
        x:    [B, L, d_model]
        mask: [1, 1, L, L]
        """
        # Pre-LN + Masked MHA + Residual
        h = x + self.mha(self.ln1(x), mask)   # [B, L, d_model]
        # Pre-LN + FFN + Residual
        out = h + self.ffn(self.ln2(h))       # [B, L, d_model]
        return out


class TinyGPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 ffn_hidden, max_seq_len, n_layers):
        super().__init__()

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ffn_hidden)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights with token embeddings (like GPT-2)
        self.lm_head.weight = self.token_embed.weight

        # Precompute causal mask and store as buffer (no grad)
        self.register_buffer("mask", causal_mask(max_seq_len))

    def forward(self, input_ids):
        """
        input_ids: [B, L] (token IDs)
        returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        assert L <= self.max_seq_len

        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]

        tok = self.token_embed(input_ids)  # [B, L, d_model]
        pos = self.pos_embed(pos_ids)      # [1, L, d_model]
        x = tok + pos                      # [B, L, d_model]

        mask = self.mask[:, :, :L, :L]     # [1,1,L,L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)                   # [B, L, d_model]
        logits = self.lm_head(x)           # [B, L, vocab_size]
        return logits




def generate_dummy_batch(batch_size, seq_len, vocab_size):
    """
    Generate random token sequences for toy training.
    input_ids: [B, L], values in [0, vocab_size-1]
    """
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)


def main():
    model = TinyGPTDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(1, num_steps + 1):
        # 1) Random batch
        input_ids = generate_dummy_batch(batch_size, max_seq_len, vocab_size).to(device)
        labels = input_ids.clone()

        # 2) Forward
        logits = model(input_ids)  # [B, L, vocab_size]

        # Shift for next-token prediction
        logits_shifted = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        labels_shifted = labels[:, 1:].contiguous()      # [B, L-1]

        Bsz, T, V = logits_shifted.shape
        loss = F.cross_entropy(
            logits_shifted.view(Bsz * T, V),
            labels_shifted.view(Bsz * T),
        )

        # 3) Backprop + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print("Step {:4d} | loss = {:.4f}".format(step, loss.item()))

    print("Training demo finished.")


if __name__ == "__main__":
    main()
