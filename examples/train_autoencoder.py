"""
Example: Training an Autoencoder for Image Reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeprecon.models.autoencoder import Autoencoder
from deeprecon.training.trainer import ReconstructionTrainer
from deeprecon.utils.metrics import MSE
from deeprecon.config.config import load_config, AUTOENCODER_CONFIG


def create_noisy_mnist_dataset(data_dir='./data', noise_factor=0.3):
    """Create a noisy MNIST dataset for denoising task"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download MNIST dataset
    train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create noisy versions
    class NoisyMNIST(torch.utils.data.Dataset):
        def __init__(self, clean_dataset, noise_factor=0.3):
            self.clean_dataset = clean_dataset
            self.noise_factor = noise_factor
        
        def __len__(self):
            return len(self.clean_dataset)
        
        def __getitem__(self, idx):
            clean_img, _ = self.clean_dataset[idx]
            noisy_img = clean_img + torch.randn_like(clean_img) * self.noise_factor
            noisy_img = torch.clamp(noisy_img, -1, 1)
            return noisy_img.view(-1), clean_img.view(-1)  # Flatten for autoencoder
    
    train_noisy = NoisyMNIST(train_dataset, noise_factor)
    test_noisy = NoisyMNIST(test_dataset, noise_factor)
    
    return train_noisy, test_noisy


def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    config.update(**AUTOENCODER_CONFIG)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_noisy_mnist_dataset()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print("Creating model...")
    model = Autoencoder(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims
    )
    
    # Create loss function and optimizer
    criterion = MSE()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create trainer
    trainer = ReconstructionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_dir=config.experiment.log_dir
    )
    
    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_best=True,
        save_every=config.experiment.save_every,
        log_images_every=config.experiment.log_images_every
    )
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'autoencoder_final.pth')
    print("Final model saved as 'autoencoder_final.pth'")


if __name__ == "__main__":
    main()