#Copyright 2023-2025 Sebastian Raschka
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0


# GPT-2 implementation from "Build a Large Language Model from Scratch"

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        self.attention_checkpoint = x
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        self.feed_forward_checkpoint = x
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        self.transformer_checkpoint = x
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        self.embedding_checkpoint = x
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        self.transformer_checkpoint = []
        for trf in self.trf_blocks:
            self.transformer_checkpoint.append([
                trf.attention_checkpoint, 
                trf.feed_forward_checkpoint, 
                trf.transformer_checkpoint])
        x = self.final_norm(x)
        logits = self.out_head(x)
        self.logits_checkpoint = logits
        return logits
    
    @staticmethod
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.detach())
    
    def load_weights(self, params):
        self.pos_emb.weight = GPTModel.assign(self.pos_emb.weight, params["wpe.weight"])
        self.tok_emb.weight = GPTModel.assign(self.tok_emb.weight, params["wte.weight"])

        for b in range(len(self.trf_blocks)):
            q_w, k_w, v_w = torch.chunk(
                params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.weight = GPTModel.assign(
                self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = GPTModel.assign(
                self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = GPTModel.assign(
                self.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = torch.chunk(
                params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = GPTModel.assign(
                self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = GPTModel.assign(
                self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = GPTModel.assign(
                self.trf_blocks[b].att.W_value.bias, v_b)

            self.trf_blocks[b].att.out_proj.weight = GPTModel.assign(
                self.trf_blocks[b].att.out_proj.weight,
                params[f"h.{b}.attn.c_proj.weight"].T)
            self.trf_blocks[b].att.out_proj.bias = GPTModel.assign(
                self.trf_blocks[b].att.out_proj.bias,
                params[f"h.{b}.attn.c_proj.bias"])

            self.trf_blocks[b].ff.layers[0].weight = GPTModel.assign(
                self.trf_blocks[b].ff.layers[0].weight,
                params[f"h.{b}.mlp.c_fc.weight"].T)
            self.trf_blocks[b].ff.layers[0].bias = GPTModel.assign(
                self.trf_blocks[b].ff.layers[0].bias,
                params[f"h.{b}.mlp.c_fc.bias"])
            self.trf_blocks[b].ff.layers[2].weight = GPTModel.assign(
                self.trf_blocks[b].ff.layers[2].weight,
                params[f"h.{b}.mlp.c_proj.weight"].T)
            self.trf_blocks[b].ff.layers[2].bias = GPTModel.assign(
                self.trf_blocks[b].ff.layers[2].bias,
                params[f"h.{b}.mlp.c_proj.bias"])

            self.trf_blocks[b].norm1.scale = GPTModel.assign(
                self.trf_blocks[b].norm1.scale,
                params[f"h.{b}.ln_1.weight"])
            self.trf_blocks[b].norm1.shift = GPTModel.assign(
                self.trf_blocks[b].norm1.shift,
                params[f"h.{b}.ln_1.bias"])
            self.trf_blocks[b].norm2.scale = GPTModel.assign(
                self.trf_blocks[b].norm2.scale,
                params[f"h.{b}.ln_2.weight"])
            self.trf_blocks[b].norm2.shift = GPTModel.assign(
                self.trf_blocks[b].norm2.shift,
                params[f"h.{b}.ln_2.bias"])

        self.final_norm.scale = GPTModel.assign(self.final_norm.scale, params["ln_f.weight"])
        self.final_norm.shift = GPTModel.assign(self.final_norm.shift, params["ln_f.bias"])
        self.out_head.weight = GPTModel.assign(self.out_head.weight, params["wte.weight"])


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

