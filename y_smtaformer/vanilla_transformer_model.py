import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VanillaTransformer(nn.Module):
    """
    Standard Transformer Encoder for Time-Series Classification.
    Fuses static features via concatenation before the final classifier.
    """
    def __init__(
        self,
        static_dim,
        dynamic_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # 1. Dynamic Feature Embedding
        self.input_projection = nn.Linear(dynamic_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 3. Static Feature Embedding
        self.static_projection = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Final Classifier
        # Input: Pooled Dynamic (d_model) + Static (d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, static, dynamic, mask=None):
        """
        static: [Batch, Static_Dim]
        dynamic: [Batch, Seq_Len, Dynamic_Dim]
        mask: Optional [Batch, Seq_Len] (padding mask, not used in this simple version)
        """
        # A. Process Dynamic Data
        # Project -> Pos Encode -> Transform
        x = self.input_projection(dynamic)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # x shape: [Batch, Seq, d_model]
        x_trans = self.transformer_encoder(x)
        
        # Global Average Pooling (collapse time dimension)
        x_pooled = x_trans.mean(dim=1) 
        
        # B. Process Static Data
        x_static = self.static_projection(static)
        
        # C. Concatenate & Classify
        combined = torch.cat([x_pooled, x_static], dim=1)
        logits = self.classifier(combined)
        
        return logits
