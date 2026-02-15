# zunel/bottleneck.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class MultiHeadAttention(nn.Module):
    def __init__(self, n_units, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_units = n_units
        self.head_dim = n_units // n_heads
        
        assert n_units % n_heads == 0
        
        self.query = nn.Linear(n_units, n_units)
        self.key = nn.Linear(n_units, n_units)
        self.value = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, n_units)
        
    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)
        
        Q = self.query(queries).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(keys).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(values).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_units)
        output = self.out(context)
        return output, attention





class GlobalStyleToken(nn.Module):
    def __init__(self, embedding_dim=256, n_style_tokens=10, n_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_style_tokens = n_style_tokens
        
        self.style_tokens = nn.Parameter(torch.randn(1, n_style_tokens, embedding_dim))
        nn.init.normal_(self.style_tokens, mean=0, std=0.5)
        
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        style_tokens = self.style_tokens.expand(batch_size, -1, -1)
        
        style_embed, _ = self.attention(
            queries=style_tokens,
            keys=inputs,
            values=inputs
        )
        
        style_embed = torch.mean(style_embed, dim=1)
        style_embed = F.normalize(style_embed, p=2, dim=-1)
        return style_embed





class SegmentGSTBottleneck(nn.Module):
    def __init__(self, input_dim=256, bottleneck_dim=1024, output_dim=256, n_style_tokens=10, n_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        
        self.pre_projection = nn.Linear(input_dim, bottleneck_dim)
        self.layer_norm = nn.LayerNorm(bottleneck_dim)
        
        self.gst = GlobalStyleToken(
            embedding_dim=bottleneck_dim,
            n_style_tokens=n_style_tokens,
            n_heads=n_heads
        )
        
        self.post_projection = nn.Linear(bottleneck_dim, output_dim)
        
    def forward(self, speaker_embedding):
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(-1)
        
        x = F.relu(self.layer_norm(self.pre_projection(speaker_embedding)))
        
        x = x.unsqueeze(1)
        
        style_embedding = self.gst(x)
        
        output = self.post_projection(style_embedding)
        output = F.normalize(output, p=2, dim=-1)
        return output.unsqueeze(-1)





class VAEBottleneck(nn.Module):
    def __init__(self, input_dim=256, bottleneck_dim=1024, output_dim=256, kl_weight=0.0001):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.kl_weight = kl_weight
        
        self.encoder_mean = nn.Linear(input_dim, bottleneck_dim)
        self.encoder_logvar = nn.Linear(input_dim, bottleneck_dim)
        
        self.decoder = nn.Linear(bottleneck_dim, output_dim)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, speaker_embedding, inference=False):
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(-1)
        
        mean = self.encoder_mean(speaker_embedding)
        logvar = self.encoder_logvar(speaker_embedding)
        
        if inference:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        
        output = self.decoder(z)
        output = F.normalize(output, p=2, dim=-1)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = torch.mean(kl_loss) * self.kl_weight
        return output.unsqueeze(-1), kl_loss





class SimplexBottleneck(nn.Module):
    def __init__(self, input_dim=256, bottleneck_dim=1024, output_dim=256, n_vertices=128):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.n_vertices = n_vertices
        
        self.projection = nn.Linear(input_dim, bottleneck_dim)
        
        self.vertices = nn.Parameter(torch.randn(n_vertices, bottleneck_dim))
        nn.init.orthogonal_(self.vertices)
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.output_projection = nn.Linear(bottleneck_dim, output_dim)
        
    def forward(self, speaker_embedding):
        if speaker_embedding.dim() == 3:
            speaker_embedding = speaker_embedding.squeeze(-1)
        
        x = self.projection(speaker_embedding)
        x = F.normalize(x, p=2, dim=-1)
        
        vertices_normalized = F.normalize(self.vertices, p=2, dim=-1)
        
        similarities = torch.matmul(x, vertices_normalized.t()) / self.temperature
        
        weights = F.softmax(similarities, dim=-1)
        
        output = torch.matmul(weights, vertices_normalized)
        output = self.output_projection(output)
        output = F.normalize(output, p=2, dim=-1)
        return output.unsqueeze(-1)





class BottleneckModule(nn.Module):
    def __init__(
        self,
        input_dim=256,
        bottleneck_dim=1024,
        output_dim=256,
        bottleneck_type='segment_gst',
        n_style_tokens=10,
        n_heads=8,
        n_vertices=128,
        kl_weight=0.0001
    ):
        super().__init__()
        self.bottleneck_type = bottleneck_type
        
        if bottleneck_type == 'segment_gst':
            self.bottleneck = SegmentGSTBottleneck(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                output_dim=output_dim,
                n_style_tokens=n_style_tokens,
                n_heads=n_heads
            )
        elif bottleneck_type == 'vae':
            self.bottleneck = VAEBottleneck(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                output_dim=output_dim,
                kl_weight=kl_weight
            )
        elif bottleneck_type == 'simplex':
            self.bottleneck = SimplexBottleneck(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                output_dim=output_dim,
                n_vertices=n_vertices
            )
        else:
            raise ValueError(f"Unknown bottleneck type: {bottleneck_type}")
    
    def forward(self, speaker_embedding, inference=False):
        if self.bottleneck_type == 'vae':
            return self.bottleneck(speaker_embedding, inference=inference)
        else:
            return self.bottleneck(speaker_embedding), None