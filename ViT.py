import torch.nn as nn
import torch
import math

class Config:
    patch_size = 16
    hidden_size = 48
    num_hidden_layers = 4
    num_attention_heads = 4
    intermediate_size = 4 * 48
    image_size = 224
    num_classes = 3
    num_channels = 3

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels, 
            config.hidden_size, 
            kernel_size=config.patch_size, 
            stride=config.patch_size)
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1,1,config.hidden_size))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches+1, config.hidden_size)
        )
    
    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embeddings
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        score = (query @ key.transpose(1,2)) / math.sqrt(key.size(-1))
        weight = torch.softmax(score, dim=-1)
        attention = weight @ value
        return attention, weight


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size)
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size) 
    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output,
                                      _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, 
                           attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs) 



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, output_attention=False):
        attention, weight = self.attention(self.layernorm1(x), output_attention)
        x = x + attention
        mlp_out = self.mlp(self.layernorm2(x))
        x = x + mlp_out
        if not output_attention:
            return (x, None)
        return (x, weight)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attention=False):
        all_attention = []
        for block in self.blocks:
            x, weight = block(x, output_attention)
            if output_attention:
                all_attention.append(weight)
        if not output_attention:
            return (x, None)
        return (x, all_attention)


class ViTClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x, output_attention=False):
        embed = self.embedding(x)
        out, all_attention = self.encoder(embed, output_attention)
        logits = self.classifier(out[:, 0, :]) #cls token only
        if not output_attention:
            return (logits, None)
        return (logits, all_attention)

        