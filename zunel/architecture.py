# zunel/architecture.py
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from zunel import layers
from zunel import helpers
from zunel import attention
from zunel.helpers import initialize_weights, compute_padding





class PhoneticEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attention.TransformerEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = torch.transpose(self.emb(x) * math.sqrt(self.hidden_channels), 1, -1)
        x_mask = torch.unsqueeze(helpers.length_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask





class TemporalPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = layers.ChannelNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = layers.ChannelNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask





class StochasticTemporalPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = layers.LogLayer()
        self.flows = nn.ModuleList()
        self.flows.append(layers.AffineLayer(2))
        for i in range(n_flows):
            self.flows.append(layers.SplineFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(layers.FlipLayer())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = layers.DepthSepConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(layers.AffineLayer(2))
        
        for i in range(4):
            self.post_flows.append(layers.SplineFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(layers.FlipLayer())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = layers.DepthSepConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.proj(self.convs(x, x_mask)) * x_mask

        if not reverse:
            assert w is not None
            logdet_tot_q = 0
            h_w = self.post_proj(self.post_convs(self.post_pre(w), x_mask)) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot += logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq
        else:
            active_flows = list(reversed(self.flows))
            active_flows = active_flows[:-2] + [active_flows[-1]]
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            
            for flow in active_flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            return torch.split(z, [1, 1], 1)[0]





class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = layers.WaveNetBlock(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(helpers.length_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.enc(self.pre(x) * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask





class WaveDecoder(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        ResBlock = layers.ResidualStack1 if resblock == "1" else layers.ResidualStack2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2,
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(initialize_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        i = 0
        while i < self.num_upsamples:
            x = self.ups[i](F.leaky_relu(x, layers.LRELU_SLOPE))

            xs = None
            j = 0
            while j < self.num_kernels:
                idx = i * self.num_kernels + j
                res_out = self.resblocks[idx](x)

                if xs is None:
                    xs = res_out
                else:
                    xs = xs + res_out
                j = j + 1
            x = xs / self.num_kernels
            i = i + 1

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print("[zunel] Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        
        for l in self.resblocks:
            l.remove_weight_norm()





class SpeakerEmbedder(nn.Module):
    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_filters)
        filter_seq = [1] + ref_filters
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(filter_seq[i], filter_seq[i + 1], (3, 3), (2, 2), (1, 1)))
            for i in range(K)
        ])
        out_channels = self._calc_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(ref_filters[-1] * out_channels, 256 // 2, batch_first=True)
        self.proj = nn.Linear(128, gin_channels)
        self.layernorm = nn.LayerNorm(self.spec_channels) if layernorm else None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)
        
        if self.layernorm is not None:
            out = self.layernorm(out)
        
        for conv in self.convs:
            out = F.relu(conv(out))
        
        out = out.transpose(1, 2)
        T, N2 = out.size(1), out.size(0)
        out = out.contiguous().view(N2, T, -1)
        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return self.proj(out.squeeze(0))

    def _calc_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L





class NormalizingFlowBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                layers.CouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)
            )
            self.flows.append(layers.FlipLayer())

    def forward(self, x, x_mask, g=None, reverse=False):
        flow_seq = reversed(self.flows) if reverse else self.flows
        
        for flow in flow_seq:
            if not reverse:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
            else:
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x





class VoiceSynthesizer(nn.Module):
    def __init__(
        self,
        n_vocab,
        spec_channels,
        latent_channels,
        base_channels,
        expansion_channels,
        attention_heads,
        encoder_layers,
        conv_kernel,
        dropout_rate,
        resblock_type,
        resblock_kernels,
        resblock_dilations,
        decoder_strides,
        decoder_base_channels,
        decoder_kernels,
        n_speakers = 256,
        embedding_dim = 256,
        zero_g = False,
        **kwargs
    ):
        super().__init__()

        self.wave_decoder = WaveDecoder(
            latent_channels, resblock_type, resblock_kernels, resblock_dilations,
            decoder_strides, decoder_base_channels, decoder_kernels,
            gin_channels=embedding_dim,
        )
        self.var_encoder = VariationalEncoder(spec_channels, latent_channels, base_channels, 5, 1, 16, gin_channels=embedding_dim)
        self.norm_flow = NormalizingFlowBlock(latent_channels, base_channels, 5, 1, 4, gin_channels=embedding_dim)
        self.n_speakers = n_speakers

        if n_speakers == 0:
            self.speaker_embedder = SpeakerEmbedder(spec_channels, embedding_dim)
        else:
            self.enc_p = PhoneticEncoder(n_vocab, latent_channels, base_channels, expansion_channels, attention_heads, encoder_layers, conv_kernel, dropout_rate)
            self.sdp = StochasticTemporalPredictor(base_channels, 192, 3, 0.5, 4, gin_channels=embedding_dim)
            self.dp = TemporalPredictor(base_channels, 256, 3, 0.5, gin_channels=embedding_dim)
            self.emb_g = nn.Embedding(n_speakers, embedding_dim)
        self.zero_g = zero_g

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., sdp_ratio=0.2, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None

        logw = (self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio + self.dp(x, x_mask, g=g) * (1 - sdp_ratio))

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(helpers.length_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = helpers.build_alignment_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.norm_flow(z_p, y_mask, g=g, reverse=True)
        o = self.wave_decoder((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        g_src = sid_src
        g_tgt = sid_tgt

        if not self.zero_g:
            g_enc = g_src
        else:
            g_enc = torch.zeros_like(g_src)

        z, m_q, logs_q, y_mask = self.var_encoder(
            y,
            y_lengths,
            g = g_enc,
            tau = tau,
        )

        z_p = self.norm_flow(z, y_mask, g=g_src)
        z_hat = self.norm_flow(z_p, y_mask, g=g_tgt, reverse=True)

        if not self.zero_g:
            g_dec = g_tgt
        else:
            g_dec = torch.zeros_like(g_tgt)

        o_hat = self.wave_decoder(
            z_hat * y_mask,
            g = g_dec,
        )
        return o_hat, y_mask, (z, z_p, z_hat)