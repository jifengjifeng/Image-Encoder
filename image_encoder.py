"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim)
        return x.transpose(0, 2, 1, 3).astype(np.float32)

    def __call__(self, x, causal=False):
        qkv = (x @ self.qkv_weight.T + self.qkv_bias).astype(np.float32)
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(self.head_dim))
        scores = scores.astype(np.float32)

        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)

        context = (coefficients @ v).astype(np.float32)
        B, H, N, dh = context.shape
        merged = context.transpose(0, 2, 1, 3).reshape(B, N, self.width)

        return (merged @ self.out_weight.T + self.out_bias).astype(np.float32)


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        norm1_out = self.norm1(x).astype(np.float32)
        update = self.attention(norm1_out, causal=causal)
        
        residual = (x + update).astype(np.float32)
        u = self.norm2(residual).astype(np.float32)
        
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias).astype(np.float32)
        mlp_update = (hidden @ self.fc2_weight.T + self.fc2_bias).astype(np.float32)
        
        return (residual + mlp_update).astype(np.float32)


class ImageEncoder:
    def __init__(self, weights):
        for k in weights:
            weights[k] = weights[k].astype(np.float32)
            
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        B, C, H, W = images.shape
        ps = 32
        grid_h = H // ps
        grid_w = W // ps
        x = images.reshape(B, C, grid_h, ps, grid_w, ps)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(B, grid_h * grid_w, C * ps * ps).astype(np.float32)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        patch_tokens = (patches @ self.patch_weight.T).astype(np.float32)

        cls = np.tile(self.cls, (images.shape[0], 1, 1)).astype(np.float32)

        tokens = np.concatenate([cls, patch_tokens], axis=1)
        
        return (tokens + self.position).astype(np.float32)

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images)).astype(np.float32)
        for block in self.blocks:
            x = block(x)
        return x.astype(np.float32)

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        cls_token = tokens[:, 0, :].astype(np.float32)
        post_norm_cls = self.post_norm(cls_token).astype(np.float32)
        output = (post_norm_cls @ self.projection).astype(np.float32)

        return unit_vector(output).astype(np.float32)