"""
Configuration Management for DeepReconXploreAi
"""

import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    name: str = "autoencoder"
    input_dim: int = 784
    latent_dim: int = 128
    hidden_dims: List[int] = None
    n_channels: int = 1
    n_classes: int = 1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    loss_function: str = "mse"
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class DataConfig:
    """Data configuration parameters"""
    dataset_type: str = "reconstruction"
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    image_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: bool = True
    noise_level: float = 0.1


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters"""
    name: str = "reconstruction_experiment"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_images_every: int = 5
    device: str = "auto"
    seed: int = 42
    resume_from: Optional[str] = None


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for section, params in kwargs.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)


def get_default_config():
    """Get default configuration"""
    return Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        experiment=ExperimentConfig()
    )


def load_config(config_path: str = None):
    """
    Load configuration from file or return default.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Config: Configuration object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    else:
        return get_default_config()


# Predefined configurations for different tasks
AUTOENCODER_CONFIG = {
    'model': {
        'name': 'autoencoder',
        'input_dim': 784,
        'latent_dim': 128,
        'hidden_dims': [512, 256]
    },
    'training': {
        'num_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'loss_function': 'mse'
    }
}

VAE_CONFIG = {
    'model': {
        'name': 'vae',
        'input_dim': 784,
        'latent_dim': 64,
        'hidden_dims': [512, 256]
    },
    'training': {
        'num_epochs': 150,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'loss_function': 'vae'
    }
}

UNET_CONFIG = {
    'model': {
        'name': 'unet',
        'n_channels': 1,
        'n_classes': 1
    },
    'training': {
        'num_epochs': 200,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'loss_function': 'combined'
    },
    'data': {
        'image_size': 256,
        'augmentation': True
    }
}

GAN_CONFIG = {
    'model': {
        'name': 'gan',
        'n_channels': 1,
        'n_classes': 1
    },
    'training': {
        'num_epochs': 300,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'loss_function': 'gan'
    },
    'data': {
        'image_size': 256,
        'augmentation': True
    }
}