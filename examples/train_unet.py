"""
Example: Training U-Net for Image Reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeprecon.models.unet import UNet
from deeprecon.training.trainer import ReconstructionTrainer
from deeprecon.utils.metrics import CombinedLoss
from deeprecon.config.config import load_config, UNET_CONFIG
from deeprecon.data.loader import ImageReconstructionDataset


def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    config.update(**UNET_CONFIG)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets (you would replace this with your actual data paths)
    # For demonstration, we'll use a placeholder
    print("Note: Replace with actual dataset paths")
    print("Creating model...")
    
    # Create model
    model = UNet(
        n_channels=config.model.n_channels,
        n_classes=config.model.n_classes,
        bilinear=True
    )
    
    # Create loss function and optimizer
    criterion = CombinedLoss(alpha=1.0, beta=0.1, gamma=0.01)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # For demonstration, create dummy data loaders
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Create dummy noisy and clean image pairs
            clean = torch.randn(1, config.data.image_size, config.data.image_size)
            noisy = clean + torch.randn_like(clean) * 0.2
            return noisy, clean
    
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
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
    torch.save(model.state_dict(), 'unet_final.pth')
    print("Final model saved as 'unet_final.pth'")


if __name__ == "__main__":
    main()