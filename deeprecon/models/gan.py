"""
GAN-based Models for Reconstruction Tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for reconstruction GAN.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        ngf (int): Generator filter size
    """
    
    def __init__(self, input_channels=1, output_channels=1, ngf=64):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        output = self.final(d1)
        
        return output


class Discriminator(nn.Module):
    """
    Discriminator network for reconstruction GAN.
    
    Args:
        input_channels (int): Number of input channels  
        ndf (int): Discriminator filter size
    """
    
    def __init__(self, input_channels=2, ndf=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return torch.sigmoid(x)


class ReconstructionGAN(nn.Module):
    """
    Complete GAN model for reconstruction tasks.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        ngf (int): Generator filter size
        ndf (int): Discriminator filter size
    """
    
    def __init__(self, input_channels=1, output_channels=1, ngf=64, ndf=64):
        super(ReconstructionGAN, self).__init__()
        
        self.generator = Generator(input_channels, output_channels, ngf)
        self.discriminator = Discriminator(input_channels + output_channels, ndf)
        
        # Initialize weights
        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """Forward pass through generator"""
        return self.generator(x)
    
    def generate(self, x):
        """Generate reconstruction"""
        return self.generator(x)
    
    def discriminate(self, input_img, target_img):
        """Discriminate between real and fake pairs"""
        combined = torch.cat([input_img, target_img], dim=1)
        return self.discriminator(combined)


class GANLoss(nn.Module):
    """
    GAN Loss function combining adversarial and reconstruction losses.
    
    Args:
        lambda_l1 (float): Weight for L1 reconstruction loss
        use_lsgan (bool): Whether to use LSGAN loss
    """
    
    def __init__(self, lambda_l1=100.0, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        
        if use_lsgan:
            self.gan_loss = nn.MSELoss()
        else:
            self.gan_loss = nn.BCELoss()
            
        self.l1_loss = nn.L1Loss()
    
    def generator_loss(self, fake_pred, fake_img, real_img):
        """Calculate generator loss"""
        # Adversarial loss
        real_label = torch.ones_like(fake_pred)
        adv_loss = self.gan_loss(fake_pred, real_label)
        
        # L1 reconstruction loss
        l1_loss = self.l1_loss(fake_img, real_img)
        
        # Combined loss
        total_loss = adv_loss + self.lambda_l1 * l1_loss
        
        return total_loss, adv_loss, l1_loss
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Calculate discriminator loss"""
        # Real loss
        real_label = torch.ones_like(real_pred)
        real_loss = self.gan_loss(real_pred, real_label)
        
        # Fake loss
        fake_label = torch.zeros_like(fake_pred)
        fake_loss = self.gan_loss(fake_pred, fake_label)
        
        # Combined loss
        total_loss = (real_loss + fake_loss) * 0.5
        
        return total_loss, real_loss, fake_loss