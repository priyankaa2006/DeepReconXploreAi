"""
Data Loading and Dataset Classes
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from PIL import Image
import torchvision.transforms as transforms


class ReconstructionDataset(Dataset):
    """
    Generic dataset class for reconstruction tasks.
    
    Args:
        data_dir (str): Directory containing the data
        transform (callable): Optional transform to apply to samples
        target_transform (callable): Optional transform to apply to targets
        input_suffix (str): Suffix for input files
        target_suffix (str): Suffix for target files
    """
    
    def __init__(self, data_dir, transform=None, target_transform=None,
                 input_suffix='_input', target_suffix='_target'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.input_suffix = input_suffix
        self.target_suffix = target_suffix
        
        # Get all input files
        self.input_files = []
        self.target_files = []
        
        for filename in os.listdir(data_dir):
            if input_suffix in filename:
                input_path = os.path.join(data_dir, filename)
                target_filename = filename.replace(input_suffix, target_suffix)
                target_path = os.path.join(data_dir, target_filename)
                
                if os.path.exists(target_path):
                    self.input_files.append(input_path)
                    self.target_files.append(target_path)
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Load input and target
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]
        
        # Handle different file types
        if input_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_data = Image.open(input_path)
            target_data = Image.open(target_path)
        elif input_path.endswith('.npy'):
            input_data = np.load(input_path)
            target_data = np.load(target_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Apply transforms
        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            target_data = self.target_transform(target_data)
        
        return input_data, target_data


class ImageReconstructionDataset(Dataset):
    """
    Dataset for image reconstruction tasks (denoising, super-resolution, etc.).
    
    Args:
        clean_dir (str): Directory containing clean images
        noisy_dir (str): Directory containing noisy/degraded images
        transform (callable): Transform to apply to images
    """
    
    def __init__(self, clean_dir, noisy_dir=None, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        
        # Get image files
        self.image_files = [f for f in os.listdir(clean_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load clean image
        clean_path = os.path.join(self.clean_dir, filename)
        clean_image = Image.open(clean_path).convert('RGB')
        
        # Load noisy image or create it
        if self.noisy_dir:
            noisy_path = os.path.join(self.noisy_dir, filename)
            noisy_image = Image.open(noisy_path).convert('RGB')
        else:
            # Add synthetic noise
            noisy_image = self._add_noise(clean_image)
        
        # Apply transforms
        clean_tensor = self.transform(clean_image)
        noisy_tensor = self.transform(noisy_image)
        
        return noisy_tensor, clean_tensor
    
    def _add_noise(self, image, noise_level=0.1):
        """Add synthetic noise to image"""
        np_image = np.array(image) / 255.0
        noise = np.random.normal(0, noise_level, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))


class DataLoader:
    """
    Custom DataLoader wrapper with additional functionality.
    
    Args:
        dataset (Dataset): Dataset to load from
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        **kwargs: Additional arguments for PyTorch DataLoader
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, 
                 num_workers=4, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def dataset_size(self):
        return len(self.dataset)
    
    def get_sample_batch(self):
        """Get a sample batch for visualization"""
        return next(iter(self.dataloader))


def create_data_loaders(train_dataset, val_dataset=None, test_dataset=None,
                       batch_size=32, num_workers=4):
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        
    Returns:
        dict: Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader
    loaders['train'] = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    # Validation loader
    if val_dataset:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    # Test loader
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return loaders


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset (Dataset): Dataset to split
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset