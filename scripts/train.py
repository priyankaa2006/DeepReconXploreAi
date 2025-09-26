#!/usr/bin/env python3
"""
Main Training Script for DeepReconXploreAi
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeprecon.config.config import Config
from deeprecon.models.autoencoder import Autoencoder, VariationalAutoencoder
from deeprecon.models.unet import UNet
from deeprecon.models.gan import ReconstructionGAN
from deeprecon.training.trainer import ReconstructionTrainer
from deeprecon.data.loader import create_data_loaders, ImageReconstructionDataset
from deeprecon.utils.metrics import MSE, CombinedLoss
from deeprecon.utils.io import create_experiment_directory, save_config


def create_model(config):
    """Create model based on configuration"""
    model_name = config.model.name.lower()
    
    if model_name == 'autoencoder':
        return Autoencoder(
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims
        )
    elif model_name == 'vae':
        return VariationalAutoencoder(
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims
        )
    elif model_name == 'unet':
        return UNet(
            n_channels=config.model.n_channels,
            n_classes=config.model.n_classes,
            bilinear=getattr(config.model, 'bilinear', True)
        )
    elif model_name == 'gan':
        return ReconstructionGAN(
            input_channels=config.model.n_channels,
            output_channels=config.model.n_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def create_loss_function(config):
    """Create loss function based on configuration"""
    loss_name = config.training.loss_function.lower()
    
    if loss_name == 'mse':
        return MSE()
    elif loss_name == 'combined':
        return CombinedLoss()
    else:
        return MSE()  # Default


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    else:
        return torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)


def create_datasets(config):
    """Create datasets based on configuration"""
    # This is a placeholder implementation
    # In practice, you would load real datasets here
    
    print("Note: Using synthetic datasets for demonstration")
    print("Replace this with your actual dataset loading logic")
    
    # Create dummy dataset for demonstration
    import torch.utils.data as data
    
    class DummyDataset(data.Dataset):
        def __init__(self, size=1000, image_size=256):
            self.size = size
            self.image_size = image_size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            if config.model.name.lower() == 'autoencoder':
                # For autoencoder, return flattened images
                clean = torch.randn(self.image_size * self.image_size)
                noisy = clean + torch.randn_like(clean) * 0.1
                return noisy, clean
            else:
                # For image models, return 2D images
                clean = torch.randn(1, self.image_size, self.image_size)
                noisy = clean + torch.randn_like(clean) * 0.1
                return noisy, clean
    
    # Create train and validation datasets
    image_size = getattr(config.data, 'image_size', 256)
    if config.model.name.lower() == 'autoencoder':
        image_size = int(image_size ** 0.5)  # Convert to linear dimension
    
    train_dataset = DummyDataset(size=2000, image_size=image_size)
    val_dataset = DummyDataset(size=400, image_size=image_size)
    
    return train_dataset, val_dataset, None


def main():
    parser = argparse.ArgumentParser(description='Train reconstruction models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Override experiment name')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = Config.from_yaml(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config.experiment.name = args.experiment_name
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Experiment: {config.experiment.name}")
    
    # Create experiment directory
    exp_dirs = create_experiment_directory('experiments', config.experiment.name)
    
    # Save configuration
    config_save_path = os.path.join(exp_dirs['configs'], 'config.yaml')
    config.to_yaml(config_save_path)
    
    # Set random seed
    torch.manual_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.experiment.seed)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model created: {model.__class__.__name__}")
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    data_loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create loss function and optimizer
    criterion = create_loss_function(config)
    optimizer = create_optimizer(model, config)
    
    # Create trainer
    trainer = ReconstructionTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val', None),
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_dir=exp_dirs['logs']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    print(f"Training for {config.training.num_epochs} epochs")
    
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_best=True,
        save_every=config.experiment.save_every,
        log_images_every=config.experiment.log_images_every
    )
    
    print("Training completed!")
    print(f"Results saved in: {exp_dirs['root']}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(exp_dirs['models'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()