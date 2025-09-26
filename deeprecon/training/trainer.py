"""
Training Pipeline for Reconstruction Models
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..utils.metrics import PSNR, SSIM
from ..utils.visualization import plot_reconstruction


class Trainer:
    """
    Generic trainer for reconstruction models.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to use for training
        log_dir (str): Directory for logging
    """
    
    def __init__(self, model, train_loader, val_loader=None, 
                 criterion=None, optimizer=None, device=None, log_dir='logs'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
        # Set default criterion and optimizer
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
            
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Initialize logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_psnr': [],
            'val_psnr': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_psnr = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            if isinstance(outputs, tuple):  # For VAE
                recon, mu, logvar = outputs
                recon_loss = self.criterion(recon, targets)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss  # Beta-VAE
                outputs = recon
            else:
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            psnr = PSNR(outputs, targets)
            running_psnr += psnr.item() if torch.is_tensor(psnr) else psnr
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr:.2f}dB' if torch.is_tensor(psnr) else f'{psnr:.2f}dB'
            })
        
        # Calculate epoch averages
        epoch_loss = running_loss / len(self.train_loader)
        epoch_psnr = running_psnr / len(self.train_loader)
        
        return epoch_loss, epoch_psnr
    
    def validate_epoch(self):
        """Validate for one epoch"""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle VAE outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                
                # Calculate metrics
                psnr = PSNR(outputs, targets)
                ssim = SSIM(outputs, targets)
                
                running_psnr += psnr.item() if torch.is_tensor(psnr) else psnr
                running_ssim += ssim
        
        # Calculate epoch averages
        epoch_loss = running_loss / len(self.val_loader)
        epoch_psnr = running_psnr / len(self.val_loader)
        epoch_ssim = running_ssim / len(self.val_loader)
        
        return epoch_loss, epoch_psnr, epoch_ssim
    
    def train(self, num_epochs, save_best=True, save_every=None):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_best (bool): Whether to save best model
            save_every (int): Save model every N epochs
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_psnr = self.train_epoch()
            
            # Validate
            if self.val_loader:
                val_loss, val_psnr, val_ssim = self.validate_epoch()
            else:
                val_loss, val_psnr, val_ssim = None, None, None
            
            # Log metrics
            self._log_metrics(train_loss, train_psnr, val_loss, val_psnr, val_ssim)
            
            # Save best model
            if save_best and val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if save_every and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Close writer
        self.writer.close()
    
    def _log_metrics(self, train_loss, train_psnr, val_loss, val_psnr, val_ssim):
        """Log training metrics"""
        # Store in history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_psnr'].append(train_psnr)
        
        if val_loss is not None:
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_psnr'].append(val_psnr)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
        self.writer.add_scalar('PSNR/Train', train_psnr, self.current_epoch)
        
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
            self.writer.add_scalar('PSNR/Validation', val_psnr, self.current_epoch)
            self.writer.add_scalar('SSIM/Validation', val_ssim, self.current_epoch)
        
        # Print epoch summary
        print(f'Epoch {self.current_epoch + 1}:')
        print(f'  Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB')
        if val_loss is not None:
            print(f'  Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB, Val SSIM: {val_ssim:.4f}')
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        torch.save(checkpoint, os.path.join(self.log_dir, filename))
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")


class ReconstructionTrainer(Trainer):
    """
    Specialized trainer for reconstruction tasks with additional features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def log_sample_reconstructions(self, num_samples=4):
        """Log sample reconstructions to tensorboard"""
        if self.val_loader is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get a batch
            inputs, targets = next(iter(self.val_loader))
            inputs, targets = inputs[:num_samples].to(self.device), targets[:num_samples].to(self.device)
            
            # Generate reconstructions
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):  # VAE
                outputs = outputs[0]
            
            # Log to tensorboard
            self.writer.add_images('Inputs', inputs, self.current_epoch)
            self.writer.add_images('Targets', targets, self.current_epoch)
            self.writer.add_images('Reconstructions', outputs, self.current_epoch)
    
    def train(self, num_epochs, save_best=True, save_every=None, log_images_every=10):
        """Extended training with image logging"""
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_psnr = self.train_epoch()
            
            # Validate
            if self.val_loader:
                val_loss, val_psnr, val_ssim = self.validate_epoch()
            else:
                val_loss, val_psnr, val_ssim = None, None, None
            
            # Log metrics
            self._log_metrics(train_loss, train_psnr, val_loss, val_psnr, val_ssim)
            
            # Log sample images
            if (epoch + 1) % log_images_every == 0:
                self.log_sample_reconstructions()
            
            # Save best model
            if save_best and val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if save_every and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Close writer
        self.writer.close()