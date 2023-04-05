import torch
import torch.nn as nn

from transformer.Modules import MultiheadAttention


class SoftMultiAttCodebook(nn.Module):
    def __init__(self, codebook_size, embed_dim, num_heads):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_word_vec = embed_dim
        self.num_heads = num_heads
        assert self.d_word_vec % self.num_heads == 0

        self.emb_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        # att(feats, att_banks) -> token_id weights -> emb_banks
        self.att_banks = nn.Parameter(torch.randn(self.codebook_size, self.d_word_vec))

        self.attention = MultiheadAttention(temperature=(self.d_word_vec // self.num_heads) ** 0.5)

    def forward(self, ref, need_weights=False):
        """
        ref: Tensor with size (B, L, representation_dim) or (1, vocab_size, representation_dim).
        """
        ref[ref != ref] = 0
        B = ref.shape[0]

        q = ref.view(B, -1, self.num_heads, self.d_word_vec // self.num_heads)
        q = q.transpose(1, 2).contiguous()  # 1 x nH x vocab_size x dword // nH
        k = self.att_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()  # 1 x nH x codebook_size x dword // nH
        v = self.emb_banks.view(-1, self.num_heads, self.d_word_vec // self.num_heads)
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        weighted_embedding, attn = self.attention(q, k, v)
        weighted_embedding = weighted_embedding.transpose(1, 2).contiguous().view(B, -1, self.d_word_vec)

        # print(torch.sum(self.att_banks), torch.sum(self.emb_banks))
        
        if need_weights:
            return weighted_embedding, attn
        else:
            return weighted_embedding, None
