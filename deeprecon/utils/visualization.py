"""
Visualization Utilities for Reconstruction Tasks
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_reconstruction(input_img, target_img, output_img, save_path=None, 
                       title="Reconstruction Results", figsize=(15, 5)):
    """
    Plot input, target, and reconstructed images side by side.
    
    Args:
        input_img (torch.Tensor): Input (noisy/degraded) image
        target_img (torch.Tensor): Target (clean) image  
        output_img (torch.Tensor): Reconstructed image
        save_path (str): Path to save the plot
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Convert tensors to numpy
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()
    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(input_img.shape) == 4:
        input_img = input_img[0]
        target_img = target_img[0]
        output_img = output_img[0]
    
    # Handle channel dimension
    if input_img.shape[0] == 1:
        input_img = input_img.squeeze(0)
        target_img = target_img.squeeze(0)
        output_img = output_img.squeeze(0)
    elif input_img.shape[0] == 3:
        input_img = np.transpose(input_img, (1, 2, 0))
        target_img = np.transpose(target_img, (1, 2, 0))
        output_img = np.transpose(output_img, (1, 2, 0))
    
    # Normalize to [0, 1]
    def normalize_img(img):
        img_min, img_max = img.min(), img.max()
        return (img - img_min) / (img_max - img_min)
    
    input_img = normalize_img(input_img)
    target_img = normalize_img(target_img)
    output_img = normalize_img(output_img)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Input image
    axes[0].imshow(input_img, cmap='gray' if len(input_img.shape) == 2 else None)
    axes[0].set_title('Input (Degraded)')
    axes[0].axis('off')
    
    # Target image
    axes[1].imshow(target_img, cmap='gray' if len(target_img.shape) == 2 else None)
    axes[1].set_title('Target (Ground Truth)')
    axes[1].axis('off')
    
    # Reconstructed image
    axes[2].imshow(output_img, cmap='gray' if len(output_img.shape) == 2 else None)
    axes[2].set_title('Reconstructed')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(history, save_path=None, figsize=(12, 8)):
    """
    Plot training and validation curves.
    
    Args:
        history (dict): Training history containing loss and metrics
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Progress', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR curves
    if 'train_psnr' in history:
        axes[0, 1].plot(epochs, history['train_psnr'], 'b-', label='Training PSNR')
        if 'val_psnr' in history and history['val_psnr']:
            axes[0, 1].plot(epochs, history['val_psnr'], 'r-', label='Validation PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # SSIM curves
    if 'val_ssim' in history and history['val_ssim']:
        axes[1, 0].plot(epochs, history['val_ssim'], 'g-', label='Validation SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('SSIM Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'learning_rate' in history and history['learning_rate']:
        axes[1, 1].plot(epochs, history['learning_rate'], 'purple', label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_batch_reconstructions(inputs, targets, outputs, num_samples=8, 
                              save_path=None, figsize=(16, 6)):
    """
    Plot multiple reconstruction examples from a batch.
    
    Args:
        inputs (torch.Tensor): Batch of input images
        targets (torch.Tensor): Batch of target images
        outputs (torch.Tensor): Batch of reconstructed images
        num_samples (int): Number of samples to plot
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    # Convert tensors to numpy
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    
    num_samples = min(num_samples, inputs.shape[0])
    
    fig, axes = plt.subplots(3, num_samples, figsize=figsize)
    fig.suptitle('Batch Reconstruction Results', fontsize=16)
    
    for i in range(num_samples):
        # Get images
        input_img = inputs[i]
        target_img = targets[i]
        output_img = outputs[i]
        
        # Handle channel dimension
        if input_img.shape[0] == 1:
            input_img = input_img.squeeze(0)
            target_img = target_img.squeeze(0)
            output_img = output_img.squeeze(0)
        elif input_img.shape[0] == 3:
            input_img = np.transpose(input_img, (1, 2, 0))
            target_img = np.transpose(target_img, (1, 2, 0))
            output_img = np.transpose(output_img, (1, 2, 0))
        
        # Normalize
        def normalize_img(img):
            img_min, img_max = img.min(), img.max()
            return (img - img_min) / (img_max - img_min)
        
        input_img = normalize_img(input_img)
        target_img = normalize_img(target_img)
        output_img = normalize_img(output_img)
        
        # Plot
        cmap = 'gray' if len(input_img.shape) == 2 else None
        
        axes[0, i].imshow(input_img, cmap=cmap)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_img, cmap=cmap)
        axes[1, i].set_title(f'Target {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(output_img, cmap=cmap)
        axes[2, i].set_title(f'Output {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_loss_comparison(loss_histories, labels, save_path=None, figsize=(10, 6)):
    """
    Compare loss curves from multiple experiments.
    
    Args:
        loss_histories (list): List of loss history dictionaries
        labels (list): Labels for each experiment
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_histories)))
    
    for i, (history, label) in enumerate(zip(loss_histories, labels)):
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 
                color=colors[i], linestyle='-', label=f'{label} (Train)')
        
        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], 
                    color=colors[i], linestyle='--', label=f'{label} (Val)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison Across Experiments')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metric_distribution(metrics, metric_name='PSNR', save_path=None, figsize=(8, 6)):
    """
    Plot distribution of reconstruction metrics.
    
    Args:
        metrics (list): List of metric values
        metric_name (str): Name of the metric
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Histogram
    plt.hist(metrics, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    
    # Statistics
    mean_val = np.mean(metrics)
    std_val = np.std(metrics)
    median_val = np.median(metrics)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.title(f'{metric_name} Distribution (μ={mean_val:.2f}, σ={std_val:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_maps(attention_weights, input_img, save_path=None, figsize=(12, 8)):
    """
    Visualize attention maps for transformer models.
    
    Args:
        attention_weights (torch.Tensor): Attention weights
        input_img (torch.Tensor): Input image
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights[0]  # Take first batch
    if len(input_img.shape) == 4:
        input_img = input_img[0]
    
    # Handle channel dimension for input image
    if input_img.shape[0] == 1:
        input_img = input_img.squeeze(0)
    elif input_img.shape[0] == 3:
        input_img = np.transpose(input_img, (1, 2, 0))
    
    num_heads = attention_weights.shape[0]
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, (num_heads + 1) // 2 + 1, figure=fig)
    
    # Original image
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(input_img, cmap='gray' if len(input_img.shape) == 2 else None)
    ax_orig.set_title('Original Image')
    ax_orig.axis('off')
    
    # Attention maps
    for i in range(min(num_heads, 8)):  # Show max 8 heads
        row = i // 4
        col = (i % 4) + 1
        
        ax = fig.add_subplot(gs[row, col])
        
        # Reshape attention to spatial dimensions
        attn_map = attention_weights[i].mean(axis=0)  # Average over tokens
        size = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape(size, size)
        
        im = ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Head {i+1}')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Attention Maps', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_model_comparison_plot(models_results, metric='PSNR', save_path=None, figsize=(10, 6)):
    """
    Create a comparison plot for different models.
    
    Args:
        models_results (dict): Dictionary with model names as keys and results as values
        metric (str): Metric to compare
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    models = list(models_results.keys())
    scores = [models_results[model][metric] for model in models]
    
    plt.figure(figsize=figsize)
    
    bars = plt.bar(models, scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()