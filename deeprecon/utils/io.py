"""
Input/Output Utilities for Model Management
"""

import os
import json
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_model(model, filepath, include_optimizer=False, optimizer=None, 
               epoch=None, loss=None, metadata=None):
    """
    Save model with additional information.
    
    Args:
        model (nn.Module): PyTorch model
        filepath (str): Path to save the model
        include_optimizer (bool): Whether to save optimizer state
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch
        loss (float): Current loss
        metadata (dict): Additional metadata
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}),
    }
    
    if include_optimizer and optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        save_dict['optimizer_class'] = optimizer.__class__.__name__
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    if loss is not None:
        save_dict['loss'] = loss
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # Save the model
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, load_optimizer=False, optimizer=None, 
               device='cpu', strict=True):
    """
    Load model from saved checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into
        filepath (str): Path to the saved model
        load_optimizer (bool): Whether to load optimizer state
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        device (str): Device to load the model on
        strict (bool): Whether to strictly enforce key matching
        
    Returns:
        dict: Additional information from the checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model.to(device)
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    
    # Return additional information
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', None),
        'metadata': checkpoint.get('metadata', {})
    }
    
    return info


def export_to_onnx(model, filepath, input_shape, device='cpu'):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model (nn.Module): PyTorch model
        filepath (str): Path to save ONNX model
        input_shape (tuple): Input tensor shape
        device (str): Device to run export on
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX format: {filepath}")


def save_training_history(history, filepath):
    """
    Save training history to JSON file.
    
    Args:
        history (dict): Training history
        filepath (str): Path to save the history
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            serializable_history[key] = value.tolist()
        elif isinstance(value, list):
            serializable_history[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
        else:
            serializable_history[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"Training history saved to {filepath}")


def load_training_history(filepath):
    """
    Load training history from JSON file.
    
    Args:
        filepath (str): Path to the history file
        
    Returns:
        dict: Training history
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"History file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    print(f"Training history loaded from {filepath}")
    return history


def export_results(results, output_dir, format='png'):
    """
    Export reconstruction results to files.
    
    Args:
        results (dict): Dictionary containing input, target, and output images
        output_dir (str): Directory to save results
        format (str): Image format ('png', 'jpg', 'npy')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key, images in results.items():
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        # Handle batch dimension
        if len(images.shape) == 4:
            batch_size = images.shape[0]
            for i in range(batch_size):
                img = images[i]
                filename = f"{key}_{i:04d}.{format}"
                filepath = os.path.join(output_dir, filename)
                
                if format == 'npy':
                    np.save(filepath, img)
                else:
                    # Convert to PIL and save
                    if img.shape[0] == 1:  # Grayscale
                        img = img.squeeze(0)
                    elif img.shape[0] == 3:  # RGB
                        img = np.transpose(img, (1, 2, 0))
                    
                    # Normalize to [0, 255]
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    if len(img.shape) == 2:
                        Image.fromarray(img, mode='L').save(filepath)
                    else:
                        Image.fromarray(img, mode='RGB').save(filepath)
    
    print(f"Results exported to {output_dir}")


def create_model_summary(model, input_shape, save_path=None):
    """
    Create and optionally save model summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape
        save_path (str): Optional path to save summary
        
    Returns:
        str: Model summary string
    """
    from torchsummary import summary
    import io
    import sys
    
    # Capture summary output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        summary(model, input_shape)
        summary_str = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(summary_str)
        print(f"Model summary saved to {save_path}")
    
    return summary_str


def save_config(config, filepath):
    """
    Save configuration to file.
    
    Args:
        config (dict or Config): Configuration to save
        filepath (str): Path to save the config
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if hasattr(config, 'to_yaml'):
        config.to_yaml(filepath)
    else:
        # Handle dictionary config
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath):
    """
    Load configuration from file.
    
    Args:
        filepath (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    import yaml
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config


def save_predictions(predictions, targets, output_path, metrics=None):
    """
    Save model predictions along with targets and metrics.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth targets
        output_path (str): Path to save the data
        metrics (dict): Optional metrics dictionary
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensors to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Save data
    save_data = {
        'predictions': predictions,
        'targets': targets
    }
    
    if metrics is not None:
        save_data['metrics'] = metrics
    
    # Save as compressed numpy archive
    np.savez_compressed(output_path, **save_data)
    print(f"Predictions saved to {output_path}")


def load_predictions(filepath):
    """
    Load saved predictions.
    
    Args:
        filepath (str): Path to the predictions file
        
    Returns:
        dict: Dictionary containing predictions, targets, and metrics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")
    
    data = np.load(filepath)
    
    result = {
        'predictions': data['predictions'],
        'targets': data['targets']
    }
    
    if 'metrics' in data.files:
        result['metrics'] = data['metrics'].item()
    
    print(f"Predictions loaded from {filepath}")
    return result


def create_experiment_directory(base_dir, experiment_name):
    """
    Create organized directory structure for experiment.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str): Name of the experiment
        
    Returns:
        dict: Dictionary with paths to different subdirectories
    """
    exp_dir = os.path.join(base_dir, experiment_name)
    
    # Create subdirectories
    subdirs = {
        'root': exp_dir,
        'models': os.path.join(exp_dir, 'models'),
        'logs': os.path.join(exp_dir, 'logs'),
        'results': os.path.join(exp_dir, 'results'),
        'configs': os.path.join(exp_dir, 'configs'),
        'plots': os.path.join(exp_dir, 'plots')
    }
    
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return subdirs


def backup_code(source_dir, backup_dir, experiment_name):
    """
    Create a backup of the source code for reproducibility.
    
    Args:
        source_dir (str): Source code directory
        backup_dir (str): Directory to store backups
        experiment_name (str): Name of the experiment
    """
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{experiment_name}_{timestamp}"
    backup_path = os.path.join(backup_dir, backup_name)
    
    # Copy source code
    shutil.copytree(source_dir, backup_path, 
                   ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git', 'logs', 'results'))
    
    print(f"Code backup created: {backup_path}")
    return backup_path