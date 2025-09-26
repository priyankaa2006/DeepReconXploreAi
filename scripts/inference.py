#!/usr/bin/env python3
"""
Main Inference Script for DeepReconXploreAi
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeprecon.config.config import Config
from deeprecon.models.autoencoder import Autoencoder, VariationalAutoencoder
from deeprecon.models.unet import UNet
from deeprecon.models.gan import ReconstructionGAN
from deeprecon.training.inference import InferenceEngine
from deeprecon.utils.io import load_model, export_results
from deeprecon.data.loader import ImageReconstructionDataset, DataLoader


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


def load_image_data(image_path, config):
    """Load and preprocess image data"""
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Apply transform
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # For autoencoder, flatten the image
    if config.model.name.lower() == 'autoencoder':
        tensor = tensor.view(1, -1)
    
    return tensor


def create_dummy_data(config, batch_size=8):
    """Create dummy data for testing"""
    if config.model.name.lower() == 'autoencoder':
        # Create flattened dummy data
        data = torch.randn(batch_size, config.model.input_dim)
        # Add some noise
        noisy_data = data + torch.randn_like(data) * 0.1
        return noisy_data, data
    else:
        # Create image dummy data
        size = config.data.image_size
        channels = config.model.n_channels
        
        clean_data = torch.randn(batch_size, channels, size, size)
        noisy_data = clean_data + torch.randn_like(clean_data) * 0.1
        
        return noisy_data, clean_data


def main():
    parser = argparse.ArgumentParser(description='Inference with reconstruction models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image/data')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    parser.add_argument('--save-results', action='store_true',
                       help='Save inference results')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = Config.from_yaml(args.config)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model created: {model.__class__.__name__}")
    
    # Load model weights
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print(f"Loading model from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    # Create inference engine
    inference_engine = InferenceEngine(
        model=model,
        device=device,
        batch_size=args.batch_size
    )
    
    # Run benchmark if requested
    if args.benchmark:
        print("Running benchmark...")
        
        if config.model.name.lower() == 'autoencoder':
            input_shape = (args.batch_size, config.model.input_dim)
        else:
            input_shape = (args.batch_size, config.model.n_channels, 
                          config.data.image_size, config.data.image_size)
        
        benchmark_results = inference_engine.benchmark(
            input_shape=input_shape,
            num_iterations=100,
            warmup_iterations=10
        )
        
        print(f"Benchmark Results:")
        print(f"  Average inference time: {benchmark_results['avg_inference_time']*1000:.2f} ms")
        print(f"  Throughput: {benchmark_results['throughput']:.2f} samples/second")
        
        # Get model complexity
        complexity = inference_engine.get_model_complexity()
        print(f"Model Complexity:")
        print(f"  Parameters: {complexity['total_parameters']:,}")
        print(f"  Model size: {complexity['model_size_mb']:.2f} MB")
    
    # Run inference
    if args.input:
        print(f"Running inference on: {args.input}")
        
        # Load input data
        input_data = load_image_data(args.input, config)
        
        # Run inference
        results = inference_engine.predict(
            input_data,
            return_metrics=False
        )
        
        print(f"Inference completed in {results['inference_time']:.4f} seconds")
        
        # Save results if requested
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save prediction
            pred_path = os.path.join(args.output_dir, 'prediction.pt')
            torch.save(results['predictions'], pred_path)
            
            print(f"Results saved to: {args.output_dir}")
    
    else:
        print("Running inference on dummy data...")
        
        # Create dummy data
        noisy_data, clean_data = create_dummy_data(config, args.batch_size)
        noisy_data = noisy_data.to(device)
        clean_data = clean_data.to(device)
        
        # Run inference
        results = inference_engine.predict(
            noisy_data,
            return_metrics=True,
            target_data=clean_data
        )
        
        print(f"Inference completed in {results['inference_time']:.4f} seconds")
        
        if 'metrics' in results:
            print(f"Metrics:")
            print(f"  PSNR: {results['metrics']['psnr']:.2f} dB")
            print(f"  SSIM: {results['metrics']['ssim']:.4f}")
        
        # Visualize results
        if config.model.name.lower() != 'autoencoder':
            inference_engine.visualize_results(
                noisy_data[:4], results['predictions'][:4], clean_data[:4],
                num_samples=min(4, args.batch_size),
                save_path=os.path.join(args.output_dir, 'visualization.png')
            )
            print(f"Visualization saved to: {os.path.join(args.output_dir, 'visualization.png')}")
        
        # Save results if requested
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            
            export_data = {
                'inputs': noisy_data,
                'predictions': results['predictions'],
                'targets': clean_data
            }
            
            export_results(export_data, args.output_dir)
            print(f"Results exported to: {args.output_dir}")
    
    print("Inference completed successfully!")


if __name__ == '__main__':
    main()