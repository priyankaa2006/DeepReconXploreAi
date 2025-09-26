"""
Example: Inference Demo for Reconstruction Models
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeprecon.models.autoencoder import Autoencoder
from deeprecon.models.unet import UNet
from deeprecon.training.inference import InferenceEngine
from deeprecon.utils.io import load_model
from deeprecon.utils.visualization import plot_reconstruction


def load_sample_image(image_path=None, size=(256, 256)):
    """
    Load and preprocess a sample image for inference.
    
    Args:
        image_path (str): Path to image file
        size (tuple): Target size
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if image_path and os.path.exists(image_path):
        # Load real image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize(size)
    else:
        # Create synthetic test image
        print("Creating synthetic test image...")
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        X, Y = np.meshgrid(x, y)
        
        # Create interesting pattern
        Z = np.sin(X * 3) * np.cos(Y * 3) + 0.5 * np.sin(X * 6) * np.sin(Y * 6)
        Z = (Z - Z.min()) / (Z.max() - Z.min())  # Normalize to [0, 1]
        
        image = Image.fromarray((Z * 255).astype(np.uint8), mode='L')
    
    # Convert to tensor and normalize
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def add_synthetic_noise(image, noise_type='gaussian', noise_level=0.2):
    """
    Add synthetic noise to clean image.
    
    Args:
        image (torch.Tensor): Clean image
        noise_type (str): Type of noise
        noise_level (float): Noise intensity
        
    Returns:
        torch.Tensor: Noisy image
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
    elif noise_type == 'salt_pepper':
        mask = torch.rand_like(image)
        noisy = image.clone()
        noisy[mask < noise_level/2] = -1  # Salt
        noisy[mask > 1 - noise_level/2] = 1  # Pepper
    else:
        noisy = image
    
    return torch.clamp(noisy, -1, 1)


def demo_autoencoder_inference():
    """Demonstrate autoencoder inference"""
    print("=" * 50)
    print("AUTOENCODER INFERENCE DEMO")
    print("=" * 50)
    
    # Create model
    model = Autoencoder(input_dim=256*256, latent_dim=128, hidden_dims=[1024, 512, 256])
    
    # Create inference engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_engine = InferenceEngine(model, device=device)
    
    # Load sample image
    clean_image = load_sample_image(size=(256, 256))
    
    # Add noise to create reconstruction task
    noisy_image = add_synthetic_noise(clean_image, 'gaussian', 0.3)
    
    print(f"Input shape: {noisy_image.shape}")
    print(f"Using device: {device}")
    
    # Flatten for autoencoder
    noisy_flat = noisy_image.view(1, -1)
    clean_flat = clean_image.view(1, -1)
    
    # Run inference
    print("Running inference...")
    results = inference_engine.predict(
        noisy_flat, 
        return_metrics=True, 
        target_data=clean_flat
    )
    
    # Print results
    print(f"Inference time: {results['inference_time']:.4f} seconds")
    if 'metrics' in results:
        print(f"PSNR: {results['metrics']['psnr']:.2f} dB")
        print(f"SSIM: {results['metrics']['ssim']:.4f}")
    
    # Reshape for visualization
    pred_image = results['predictions'].view(1, 1, 256, 256)
    
    # Visualize results
    plot_reconstruction(
        noisy_image, clean_image, pred_image,
        title="Autoencoder Reconstruction Demo",
        save_path="autoencoder_demo.png"
    )
    
    print("Results saved to 'autoencoder_demo.png'")


def demo_unet_inference():
    """Demonstrate U-Net inference"""
    print("=" * 50)
    print("U-NET INFERENCE DEMO")
    print("=" * 50)
    
    # Create model
    model = UNet(n_channels=1, n_classes=1)
    
    # Create inference engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_engine = InferenceEngine(model, device=device)
    
    # Load sample image
    clean_image = load_sample_image(size=(256, 256))
    
    # Add noise to create reconstruction task
    noisy_image = add_synthetic_noise(clean_image, 'gaussian', 0.2)
    
    print(f"Input shape: {noisy_image.shape}")
    print(f"Using device: {device}")
    
    # Run inference
    print("Running inference...")
    results = inference_engine.predict(
        noisy_image, 
        return_metrics=True, 
        target_data=clean_image
    )
    
    # Print results
    print(f"Inference time: {results['inference_time']:.4f} seconds")
    if 'metrics' in results:
        print(f"PSNR: {results['metrics']['psnr']:.2f} dB")
        print(f"SSIM: {results['metrics']['ssim']:.4f}")
    
    # Visualize results
    plot_reconstruction(
        noisy_image, clean_image, results['predictions'].unsqueeze(0),
        title="U-Net Reconstruction Demo",
        save_path="unet_demo.png"
    )
    
    print("Results saved to 'unet_demo.png'")


def demo_batch_inference():
    """Demonstrate batch inference"""
    print("=" * 50)
    print("BATCH INFERENCE DEMO")
    print("=" * 50)
    
    # Create model
    model = UNet(n_channels=1, n_classes=1)
    
    # Create inference engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_engine = InferenceEngine(model, device=device, batch_size=4)
    
    # Create batch of sample images
    batch_size = 8
    images = []
    noisy_images = []
    
    for i in range(batch_size):
        # Create different synthetic images
        clean = load_sample_image(size=(128, 128))  # Smaller for batch processing
        noisy = add_synthetic_noise(clean, 'gaussian', 0.15 + i * 0.05)  # Varying noise
        
        images.append(clean)
        noisy_images.append(noisy)
    
    # Stack into batches
    clean_batch = torch.cat(images, dim=0)
    noisy_batch = torch.cat(noisy_images, dim=0)
    
    print(f"Batch shape: {noisy_batch.shape}")
    print(f"Using device: {device}")
    
    # Run batch inference
    print("Running batch inference...")
    results = inference_engine.predict(
        noisy_batch, 
        return_metrics=True, 
        target_data=clean_batch
    )
    
    # Print results
    print(f"Batch inference time: {results['inference_time']:.4f} seconds")
    print(f"Average time per sample: {results['inference_time']/batch_size:.4f} seconds")
    
    if 'metrics' in results:
        print(f"Average PSNR: {results['metrics']['psnr']:.2f} dB")
        print(f"Average SSIM: {results['metrics']['ssim']:.4f}")
    
    # Visualize batch results
    from deeprecon.utils.visualization import plot_batch_reconstructions
    
    plot_batch_reconstructions(
        noisy_batch[:4], clean_batch[:4], results['predictions'][:4],
        num_samples=4,
        save_path="batch_demo.png"
    )
    
    print("Batch results saved to 'batch_demo.png'")


def demo_model_benchmark():
    """Demonstrate model benchmarking"""
    print("=" * 50)
    print("MODEL BENCHMARK DEMO")
    print("=" * 50)
    
    # Create model
    model = UNet(n_channels=1, n_classes=1)
    
    # Create inference engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_engine = InferenceEngine(model, device=device)
    
    # Benchmark different input sizes
    input_shapes = [
        (1, 1, 128, 128),
        (4, 1, 128, 128),
        (1, 1, 256, 256),
        (4, 1, 256, 256)
    ]
    
    for shape in input_shapes:
        print(f"\nBenchmarking input shape: {shape}")
        results = inference_engine.benchmark(
            input_shape=shape,
            num_iterations=50,
            warmup_iterations=5
        )
        
        print(f"Average inference time: {results['avg_inference_time']*1000:.2f} ms")
        print(f"Throughput: {results['throughput']:.2f} samples/second")
    
    # Get model complexity
    complexity = inference_engine.get_model_complexity()
    print(f"\nModel Complexity:")
    print(f"Total parameters: {complexity['total_parameters']:,}")
    print(f"Trainable parameters: {complexity['trainable_parameters']:,}")
    print(f"Model size: {complexity['model_size_mb']:.2f} MB")


def main():
    """Run all inference demos"""
    print("DeepReconXploreAi - Inference Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_autoencoder_inference()
        print("\n" + "="*50 + "\n")
        
        demo_unet_inference()
        print("\n" + "="*50 + "\n")
        
        demo_batch_inference()
        print("\n" + "="*50 + "\n")
        
        demo_model_benchmark()
        
        print("\n" + "="*50)
        print("All demos completed successfully!")
        print("Check the generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        print("This is expected if you haven't trained models yet.")
        print("The demos show the inference pipeline structure.")


if __name__ == "__main__":
    main()