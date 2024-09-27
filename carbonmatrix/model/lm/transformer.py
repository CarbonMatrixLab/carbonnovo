from torch import nn

from esm import modules  as E

from carbonmatrix.model.lm.multihead_attention import MultiheadAttention
from carbonmatrix.model.common_modules import Linear

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        lora_config = {},
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings

        self.lora_config = lora_config

        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = E.ESM1bLayerNorm if use_esm1b_layer_norm else E.ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
            lora_config=self.lora_config,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, self.ffn_embed_dim, init='relu', **self.lora_config)
        self.fc2 = Linear(self.ffn_embed_dim, self.embed_dim, init='relu', **self.lora_config)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, residx=None, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            residx=residx,
            key_padding_mask=self_attn_padding_mask,
            need_weights=need_head_weights,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = E.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
