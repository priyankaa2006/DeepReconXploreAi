"""
Evaluation Metrics for Reconstruction Tasks
"""

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim


def PSNR(output, target, max_pixel=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        output (torch.Tensor): Reconstructed image
        target (torch.Tensor): Ground truth image
        max_pixel (float): Maximum pixel value
        
    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def SSIM(output, target, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        output (torch.Tensor): Reconstructed image
        target (torch.Tensor): Ground truth image
        data_range (float): Data range
        
    Returns:
        float: SSIM value
    """
    # Convert to numpy for skimage
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(output.shape) == 4:  # Batch, Channel, Height, Width
        ssim_vals = []
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                ssim_val = ssim(
                    target[i, j], 
                    output[i, j], 
                    data_range=data_range
                )
                ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    else:
        return ssim(target, output, data_range=data_range)


class MSE(nn.Module):
    """Mean Squared Error Loss"""
    
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, output, target):
        return self.mse(output, target)


class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error)"""
    
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, output, target):
        return self.l1(output, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features.
    
    Args:
        feature_layers (list): VGG layers to use for feature extraction
        use_gpu (bool): Whether to use GPU
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features
        
        self.feature_extractor = nn.Sequential()
        for i, layer in enumerate(vgg):
            self.feature_extractor.add_module(str(i), layer)
            if i in feature_layers:
                self.feature_extractor.add_module(f'hook_{i}', nn.Identity())
        
        self.feature_layers = feature_layers
        self.mse = nn.MSELoss()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        if use_gpu and torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
    
    def forward(self, output, target):
        """Calculate perceptual loss"""
        # Normalize inputs to [0, 1] if needed
        if output.max() > 1.0:
            output = output / 255.0
        if target.max() > 1.0:
            target = target / 255.0
            
        # Convert grayscale to RGB if needed
        if output.size(1) == 1:
            output = output.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Extract features
        output_features = self._extract_features(output)
        target_features = self._extract_features(target)
        
        # Calculate loss
        loss = 0
        for out_feat, tar_feat in zip(output_features, target_features):
            loss += self.mse(out_feat, tar_feat)
        
        return loss
    
    def _extract_features(self, x):
        """Extract features from specified layers"""
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class CombinedLoss(nn.Module):
    """
    Combined loss function for reconstruction tasks.
    
    Args:
        alpha (float): Weight for MSE loss
        beta (float): Weight for L1 loss
        gamma (float): Weight for perceptual loss
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse = MSE()
        self.l1 = L1Loss()
        self.perceptual = PerceptualLoss()
    
    def forward(self, output, target):
        """Calculate combined loss"""
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        perceptual_loss = self.perceptual(output, target)
        
        total_loss = (
            self.alpha * mse_loss +
            self.beta * l1_loss +
            self.gamma * perceptual_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item(),
            'total': total_loss.item()
        }