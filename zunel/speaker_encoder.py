# zunel/speaker_encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, t = x.size()
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(2)





class Res2NetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, scale=4, dilation=1):
        super().__init__()
        self.scale = scale
        width = channels // scale
        self.nums = scale - 1
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])

    def forward(self, x):
        spx = torch.split(x, x.size(1) // self.scale, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        return out





class SERes2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, scale=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.res2net = Res2NetBlock(out_channels, kernel_size, scale, dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res2net(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.shortcut:
            residual = self.shortcut(x)
        
        out += residual
        return F.relu(out)





class AttentiveStatPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, channels, 1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        return torch.cat((mu, sg), 1)





class ECAPATDNNEncoder(nn.Module):
    def __init__(
        self,
        input_size = 80,
        channels = [512, 512, 512, 512, 1536],
        embedding_dim = 256,
        kernel_sizes = [5, 3, 3, 3, 1],
        dilations = [1, 2, 3, 4, 1]
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, channels[0], kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        
        self.layer1 = SERes2Block(channels[0], channels[1], kernel_sizes[1], dilations[1])
        self.layer2 = SERes2Block(channels[1], channels[2], kernel_sizes[2], dilations[2])
        self.layer3 = SERes2Block(channels[2], channels[3], kernel_sizes[3], dilations[3])
        
        self.conv2 = nn.Conv1d(channels[3], channels[4], 1)
        self.bn2 = nn.BatchNorm1d(channels[4])
        
        self.pooling = AttentiveStatPooling(channels[4])
        self.bn3 = nn.BatchNorm1d(channels[4] * 2)
        self.fc = nn.Linear(channels[4] * 2, embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x = F.relu(self.bn2(self.conv2(x3)))
        x = torch.cat([x, x1, x2, x3], dim=1)
        
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.fc(x)
        return x





class ImprovedSpeakerEmbedder(nn.Module):
    def __init__(self, spec_channels, embedding_dim=256):
        super().__init__()
        self.spec_channels = spec_channels
        self.embedding_dim = embedding_dim
        self.encoder = ECAPATDNNEncoder(
            input_size = spec_channels,
            embedding_dim = embedding_dim
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        if x.size(1) != self.spec_channels:
            x = x.transpose(1, 2)
        
        emb = self.encoder(x)
        emb = F.normalize(emb, p=2, dim=1)
        return emb





class MultiSampleEmbeddingAggregator(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings):
        weights = F.softmax(self.attention_net(embeddings).squeeze(-1), dim=0)
        aggregated = torch.sum(embeddings * weights.unsqueeze(-1), dim=0)
        return F.normalize(aggregated, p=2, dim=-1)