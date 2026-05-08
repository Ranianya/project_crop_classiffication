# model/mctnet.py
import torch
import torch.nn as nn
import numpy as np

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        return x * self.sigmoid(y)


class ALPE(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.eca = ECALayer(d_model)
        
    def create_positional_encoding(self, time_steps):
        pe = torch.zeros(time_steps, self.d_model)
        position = torch.arange(0, time_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x, mask):
        batch_size, time_steps, d_model = x.shape
        pe = self.create_positional_encoding(time_steps).to(x.device)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        mask_mean = mask.mean(dim=-1, keepdim=True)
        mask_binary = (mask_mean > 0.5).float()
        pe_masked = pe * mask_binary
        pe_conv = pe_masked.transpose(1, 2)
        pe_conv = self.conv1d(pe_conv)
        alpe = self.eca(pe_conv)
        return alpe.transpose(1, 2)


class CNNSubmodule(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, kernel_sizes=[3, 5, 7], dropout=0.5):
        super().__init__()
        num_scales = len(kernel_sizes)
        if hidden_channels is None:
            hidden_channels = in_channels
        while hidden_channels % num_scales != 0:
            hidden_channels += 1
        self.kernel_sizes = kernel_sizes
        self.num_scales = num_scales
        self.hidden_channels = hidden_channels
        out_channels_per_conv = hidden_channels // num_scales
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv1d(in_channels, out_channels_per_conv, 
                         kernel_size=k, padding=k//2)
            )
        self.final_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        x_t = x.transpose(1, 2)
        multi_scale_out = []
        for conv in self.convs:
            out = conv(x_t)
            multi_scale_out.append(out)
        out = torch.cat(multi_scale_out, dim=1)
        out = self.final_conv(out)
        out = self.bn(out)
        out = self.dropout(out)
        skip = self.skip_proj(x_t)
        out = out + skip
        out = self.relu(out)
        return out.transpose(1, 2)


class TransformerSubmodule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.5, use_alpe=False):
        super().__init__()
        self.use_alpe = use_alpe
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        if use_alpe:
            self.alpe = ALPE(d_model)
        
    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            alpe = self.alpe(x, mask)
            x = x + alpe
        if mask is not None:
            mask_float = mask.float()
            mask_mean = mask_float.mean(dim=-1)
            key_padding_mask = (mask_mean < 0.5)
        else:
            key_padding_mask = None
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class CTFusion(nn.Module):
    def __init__(self, d_model, nhead, kernel_sizes=[3, 5, 7], use_alpe=False, dropout=0.5):
        super().__init__()
        self.cnn = CNNSubmodule(d_model, hidden_channels=d_model*2, kernel_sizes=kernel_sizes, dropout=dropout)
        self.transformer = TransformerSubmodule(d_model, nhead, use_alpe=use_alpe, dropout=dropout)
        
    def forward(self, x, mask=None):
        cnn_out = self.cnn(x)
        if self.transformer.use_alpe:
            trans_out, attn_weights = self.transformer(x, mask)
        else:
            trans_out, attn_weights = self.transformer(x)
        out = cnn_out + trans_out
        return out, attn_weights


class MCTNet(nn.Module):
    def __init__(self, input_channels=13, time_steps=36, d_model=64, nhead=4,
                 n_stages=3, n_classes=5, kernel_sizes=[3, 5, 7], dropout=0.5,
                 covariate_dim=0):
        super().__init__()
        assert d_model % nhead == 0
        self.n_stages = n_stages
        self.d_model = d_model
        self.covariate_dim = covariate_dim
        
        self.input_proj = nn.Linear(input_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            use_alpe = (i == 0)
            self.stages.append(CTFusion(d_model, nhead, kernel_sizes, use_alpe, dropout))
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        if covariate_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(d_model + covariate_dim, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes)
            )
        
    def forward(self, x, mask, covariates=None):
        x = self.input_proj(x)
        x = self.dropout(x)
        current_time = x.shape[1]
        
        for i, stage in enumerate(self.stages):
            if i == 0:
                x, attn = stage(x, mask)
            else:
                x, attn = stage(x)
            
            if i < self.n_stages - 1:
                if current_time % 2 == 0:
                    x = x.transpose(1, 2)
                    x = self.pool(x)
                    x = x.transpose(1, 2)
                    current_time = x.shape[1]
        
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        if covariates is not None:
            x = torch.cat([x, covariates], dim=1)
        
        logits = self.classifier(x)
        return logits, attn