"""
Inference Engine for Reconstruction Models
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from ..utils.metrics import PSNR, SSIM
from ..utils.visualization import plot_reconstruction, plot_batch_reconstructions


class InferenceEngine:
    """
    Inference engine for reconstruction models.
    
    Args:
        model (nn.Module): Trained model
        device (str): Device for inference
        batch_size (int): Batch size for inference
    """
    
    def __init__(self, model, device='cpu', batch_size=32):
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Statistics
        self.inference_stats = {
            'total_samples': 0,
            'total_time': 0,
            'avg_time_per_sample': 0,
            'throughput': 0
        }
    
    def predict(self, input_data, return_metrics=False, target_data=None):
        """
        Perform inference on input data.
        
        Args:
            input_data (torch.Tensor or np.ndarray): Input data
            return_metrics (bool): Whether to compute and return metrics
            target_data (torch.Tensor or np.ndarray): Target data for metrics
            
        Returns:
            dict: Predictions and optionally metrics
        """
        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).float()
        
        if target_data is not None and isinstance(target_data, np.ndarray):
            target_data = torch.from_numpy(target_data).float()
        
        # Move to device
        input_data = input_data.to(self.device)
        if target_data is not None:
            target_data = target_data.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Handle batch dimension
            if len(input_data.shape) == 3:  # Single image
                input_data = input_data.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False
            
            # Batch processing for large inputs
            predictions = []
            num_batches = (input_data.size(0) + self.batch_size - 1) // self.batch_size
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, input_data.size(0))
                
                batch_input = input_data[start_idx:end_idx]
                batch_pred = self.model(batch_input)
                
                # Handle VAE outputs
                if isinstance(batch_pred, tuple):
                    batch_pred = batch_pred[0]
                
                predictions.append(batch_pred)
            
            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=0)
            
            if single_sample:
                predictions = predictions.squeeze(0)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Update statistics
        self._update_stats(input_data.size(0), inference_time)
        
        # Prepare results
        results = {
            'predictions': predictions,
            'inference_time': inference_time
        }
        
        # Compute metrics if requested
        if return_metrics and target_data is not None:
            metrics = self._compute_metrics(predictions, target_data)
            results['metrics'] = metrics
        
        return results
    
    def predict_dataloader(self, dataloader, save_results=False, output_dir=None):
        """
        Perform inference on entire dataloader.
        
        Args:
            dataloader (DataLoader): Data loader for inference
            save_results (bool): Whether to save results
            output_dir (str): Directory to save results
            
        Returns:
            dict: Aggregated results and metrics
        """
        all_predictions = []
        all_targets = []
        all_inputs = []
        total_time = 0
        
        print(f"Running inference on {len(dataloader)} batches...")
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Inference")):
            # Run inference
            results = self.predict(inputs, return_metrics=False)
            
            # Store results
            all_inputs.append(inputs)
            all_predictions.append(results['predictions'])
            all_targets.append(targets)
            total_time += results['inference_time']
        
        # Concatenate all results
        all_inputs = torch.cat(all_inputs, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute overall metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        # Aggregate results
        results = {
            'inputs': all_inputs,
            'predictions': all_predictions,
            'targets': all_targets,
            'metrics': metrics,
            'total_inference_time': total_time,
            'avg_time_per_batch': total_time / len(dataloader)
        }
        
        # Save results if requested
        if save_results and output_dir:
            self._save_inference_results(results, output_dir)
        
        print(f"Inference completed in {total_time:.2f} seconds")
        print(f"Average PSNR: {metrics['psnr']:.2f} dB")
        print(f"Average SSIM: {metrics['ssim']:.4f}")
        
        return results
    
    def benchmark(self, input_shape, num_iterations=100, warmup_iterations=10):
        """
        Benchmark model performance.
        
        Args:
            input_shape (tuple): Shape of input tensor
            num_iterations (int): Number of inference iterations
            warmup_iterations (int): Number of warmup iterations
            
        Returns:
            dict: Benchmark results
        """
        print(f"Benchmarking model with input shape {input_shape}...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        print(f"Running {num_iterations} iterations...")
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        start_time = time.time()
        
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        avg_time = total_time / num_iterations
        throughput = input_shape[0] / avg_time  # samples per second
        
        results = {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'num_iterations': num_iterations,
            'input_shape': input_shape
        }
        
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/second")
        
        return results
    
    def visualize_results(self, inputs, predictions, targets, num_samples=8, 
                         save_path=None):
        """
        Visualize inference results.
        
        Args:
            inputs (torch.Tensor): Input images
            predictions (torch.Tensor): Predicted images
            targets (torch.Tensor): Target images
            num_samples (int): Number of samples to visualize
            save_path (str): Path to save visualization
        """
        # Select random samples
        indices = torch.randperm(inputs.size(0))[:num_samples]
        
        sample_inputs = inputs[indices]
        sample_predictions = predictions[indices]
        sample_targets = targets[indices]
        
        # Create visualization
        plot_batch_reconstructions(
            sample_inputs, sample_targets, sample_predictions,
            num_samples=num_samples, save_path=save_path
        )
    
    def _compute_metrics(self, predictions, targets):
        """Compute reconstruction metrics"""
        # Move to CPU for metric computation
        pred_cpu = predictions.detach().cpu()
        target_cpu = targets.detach().cpu()
        
        # Compute PSNR
        psnr_values = []
        ssim_values = []
        
        for i in range(pred_cpu.size(0)):
            pred_img = pred_cpu[i]
            target_img = target_cpu[i]
            
            # PSNR
            psnr = PSNR(pred_img, target_img)
            if torch.is_tensor(psnr):
                psnr = psnr.item()
            psnr_values.append(psnr)
            
            # SSIM
            ssim = SSIM(pred_img, target_img)
            ssim_values.append(ssim)
        
        return {
            'psnr': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'psnr_values': psnr_values,
            'ssim_values': ssim_values
        }
    
    def _update_stats(self, num_samples, inference_time):
        """Update inference statistics"""
        self.inference_stats['total_samples'] += num_samples
        self.inference_stats['total_time'] += inference_time
        
        if self.inference_stats['total_samples'] > 0:
            self.inference_stats['avg_time_per_sample'] = (
                self.inference_stats['total_time'] / self.inference_stats['total_samples']
            )
            self.inference_stats['throughput'] = (
                self.inference_stats['total_samples'] / self.inference_stats['total_time']
            )
    
    def _save_inference_results(self, results, output_dir):
        """Save inference results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions and targets
        torch.save(results['predictions'], output_path / 'predictions.pt')
        torch.save(results['targets'], output_path / 'targets.pt')
        torch.save(results['inputs'], output_path / 'inputs.pt')
        
        # Save metrics
        import json
        with open(output_path / 'metrics.json', 'w') as f:
            # Make metrics JSON serializable
            serializable_metrics = {}
            for key, value in results['metrics'].items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    serializable_metrics[key] = list(value)
                else:
                    serializable_metrics[key] = float(value)
            json.dump(serializable_metrics, f, indent=2)
        
        # Create visualizations
        self.visualize_results(
            results['inputs'][:16], 
            results['predictions'][:16], 
            results['targets'][:16],
            save_path=str(output_path / 'visualization.png')
        )
        
        print(f"Results saved to {output_dir}")
    
    def get_model_complexity(self):
        """
        Calculate model complexity metrics.
        
        Returns:
            dict: Model complexity information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def export_predictions(self, predictions, targets=None, output_format='numpy'):
        """
        Export predictions in different formats.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Optional target data
            output_format (str): Export format ('numpy', 'images', 'csv')
            
        Returns:
            dict: Exported data in specified format
        """
        if output_format == 'numpy':
            result = {'predictions': predictions.detach().cpu().numpy()}
            if targets is not None:
                result['targets'] = targets.detach().cpu().numpy()
            return result
        
        elif output_format == 'images':
            # Convert tensors to PIL images
            from torchvision.transforms.functional import to_pil_image
            
            images = []
            for i in range(predictions.size(0)):
                # Denormalize if needed
                pred = predictions[i]
                if pred.min() < 0:  # Assume [-1, 1] normalization
                    pred = (pred + 1) / 2
                
                img = to_pil_image(torch.clamp(pred, 0, 1))
                images.append(img)
            
            return {'images': images}
        
        elif output_format == 'csv':
            # Flatten predictions for CSV export
            import pandas as pd
            
            pred_flat = predictions.view(predictions.size(0), -1).detach().cpu().numpy()
            df = pd.DataFrame(pred_flat)
            
            if targets is not None:
                target_flat = targets.view(targets.size(0), -1).detach().cpu().numpy()
                # Add targets as additional columns
                target_df = pd.DataFrame(target_flat, columns=[f'target_{i}' for i in range(target_flat.shape[1])])
                df = pd.concat([df, target_df], axis=1)
            
            return {'dataframe': df}
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


class BatchInferenceEngine(InferenceEngine):
    """
    Optimized inference engine for large-scale batch processing.
    """
    
    def __init__(self, model, device='cpu', batch_size=32, use_amp=False):
        super().__init__(model, device, batch_size)
        self.use_amp = use_amp and device != 'cpu'
        
        if self.use_amp:
            from torch.cuda.amp import autocast
            self.autocast = autocast
    
    def predict(self, input_data, return_metrics=False, target_data=None):
        """Optimized prediction with optional mixed precision"""
        if self.use_amp:
            with self.autocast():
                return super().predict(input_data, return_metrics, target_data)
        else:
            return super().predict(input_data, return_metrics, target_data)