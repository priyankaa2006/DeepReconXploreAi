"""
Transformer-based Models for Reconstruction Tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image reconstruction tasks.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        mlp_ratio (int): MLP expansion ratio
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4):
        super(VisionTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * in_channels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        
        # Remove class token
        x = x[:, 1:]
        
        # Reconstruction
        x = self.reconstruction_head(x)
        
        # Reshape to image
        x = x.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C, H, W)
        
        return x


class ReconstructionTransformer(nn.Module):
    """
    Transformer model specifically designed for reconstruction tasks.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        num_layers (int): Number of layers
        num_heads (int): Number of attention heads
        output_dim (int): Output dimension
    """
    
    def __init__(self, input_dim=784, hidden_dim=512, num_layers=6, 
                 num_heads=8, output_dim=784):
        super(ReconstructionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x.unsqueeze(1))  # Add sequence dimension
        
        # Positional encoding
        x = self.pos_encoding(x.transpose(0, 1))
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.output_proj(x.transpose(0, 1).squeeze(1))
        
        return x


class TransformerUNet(nn.Module):
    """
    U-Net with Transformer blocks for better long-range dependencies.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        embed_dims (list): Embedding dimensions for each stage
        num_heads (list): Number of heads for each stage
    """
    
    def __init__(self, in_channels=1, out_channels=1, 
                 embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16]):
        super(TransformerUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = nn.ModuleList()
        prev_dim = in_channels
        
        for i, (dim, heads) in enumerate(zip(embed_dims, num_heads)):
            if i == 0:
                # First layer: Conv + Transformer
                layer = nn.Sequential(
                    nn.Conv2d(prev_dim, dim, 3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                    TransformerBlock2D(dim, heads)
                )
            else:
                # Other layers: Downsample + Conv + Transformer
                layer = nn.Sequential(
                    nn.Conv2d(prev_dim, dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                    TransformerBlock2D(dim, heads)
                )
            
            self.encoder.append(layer)
            prev_dim = dim
        
        # Decoder
        self.decoder = nn.ModuleList()
        
        for i, (dim, heads) in enumerate(zip(reversed(embed_dims[:-1]), reversed(num_heads[:-1]))):
            layer = nn.Sequential(
                nn.ConvTranspose2d(prev_dim, dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                TransformerBlock2D(dim * 2, heads)  # *2 for skip connection
            )
            self.decoder.append(layer)
            prev_dim = dim
        
        # Final output
        self.final = nn.Conv2d(embed_dims[0], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:  # Don't store last layer
                skip_connections.append(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer[0](x)  # Upsample
            x = layer[1](x)  # BatchNorm
            x = layer[2](x)  # ReLU
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            if x.shape != skip.shape:
                # Handle size mismatch
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = layer[3](x)  # Transformer
        
        # Final output
        x = self.final(x)
        
        return x


class TransformerBlock2D(nn.Module):
    """2D Transformer block for image processing"""
    
    def __init__(self, channels, num_heads, mlp_ratio=4):
        super(TransformerBlock2D, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP
        x_norm = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm)
        x_flat = x_flat + mlp_out
        
        # Reshape back
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        
        return x