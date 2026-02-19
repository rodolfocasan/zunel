# zunel/core/adapters.py
import torch
import torch.nn as nn
import torch.nn.functional as F





class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_embed_dim, adapter_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_embed_dim = bottleneck_embed_dim
        self.adapter_dim = adapter_dim
        
        combined_dim = input_dim + bottleneck_embed_dim
        
        self.down_project = nn.Linear(combined_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.xavier_uniform_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, layer_output, bottleneck_embedding):
        if layer_output.dim() == 3:
            B, C, T = layer_output.size()
            layer_output_flat = layer_output.transpose(1, 2).contiguous().view(B * T, C)
        else:
            layer_output_flat = layer_output
            B, C = layer_output.size()
            T = 1
        
        if bottleneck_embedding.dim() == 3:
            bottleneck_embedding = bottleneck_embedding.squeeze(-1)
        
        bottleneck_expanded = bottleneck_embedding.unsqueeze(1).expand(B, T, self.bottleneck_embed_dim)
        bottleneck_flat = bottleneck_expanded.contiguous().view(B * T, self.bottleneck_embed_dim)
        
        combined = torch.cat([layer_output_flat, bottleneck_flat], dim=-1)
        
        adapter_output = self.down_project(combined)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.up_project(adapter_output)
        
        output = layer_output_flat + adapter_output
        
        output = self.layer_norm(output)
        
        if T > 1:
            output = output.view(B, T, C).transpose(1, 2).contiguous()
        else:
            output = output.view(B, C)
        return output





class AdapterWrapper(nn.Module):
    def __init__(self, base_module, adapter):
        super().__init__()
        self.base_module = base_module
        self.adapter = adapter
        
    def forward(self, x, bottleneck_embedding=None, **kwargs):
        output = self.base_module(x, **kwargs)
        
        if bottleneck_embedding is not None and self.adapter is not None:
            output = self.adapter(output, bottleneck_embedding)
        return output





class SpeakerAdapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, speaker_embedding):
        adapted = self.projection(speaker_embedding)
        return speaker_embedding + adapted