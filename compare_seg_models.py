#!/usr/bin/env python3
"""
Comprehensive Segmentation Model Comparison for Person Silhouette Generation

This script benchmarks 5 segmentation models on video frames:
1. U2Net - Salient object detection
2. SAM (Segment Anything Model) - Universal segmentation
3. FastSAM - Fast version of SAM
4. YOLO11n-seg - Object detection + segmentation
5. HRNet - High-resolution semantic segmentation

Metrics evaluated:
- Inference speed (FPS)
- Memory usage
- Silhouette quality (IoU, Dice coefficient)
- Processing time per frame

Usage:
    python compare_seg_models.py --video path/to/video.mp4 --max_frames 50
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms

# Import local models
import models.u2net as u2net_module
import models.hrnet as hrnet_module
from ultralytics import YOLO, SAM, FastSAM
import config


class SegmentationModelComparison:
    """
    Comprehensive comparison of segmentation models for person silhouette generation
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the comparison framework
        
        Args:
            device: 'cuda', 'mps', 'cpu', or 'auto' for automatic selection
        """
        self.device = self._setup_device(device)
        self.models = {}
        self.results = defaultdict(list)
        # Will store per-frame person boxes from YOLO pass to guide SAM/FastSAM
        self.person_bboxes_per_frame: Optional[List[List[List[float]]]] = None
        # Will store per-frame unified person mask from YOLO to restrict other models
        self.person_masks_per_frame: Optional[List[Optional[np.ndarray]]] = None
        
        # Set MPS fallback if needed
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print(f"🚀 Initialized comparison framework on device: {self.device}")
        
        # Verify device availability
        if self.device == 'mps' and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available, falling back to CPU")
            self.device = 'cpu'
        elif self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
    
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def load_model(self, model_name: str) -> bool:
        """Load a single model by name"""
        try:
            if model_name == 'u2net':
                self.models[model_name] = self._load_u2net()
            elif model_name == 'hrnet':
                self.models[model_name] = self._load_hrnet()
            elif model_name == 'yolo11n_seg':
                model = YOLO('../weights/yolo11n-seg.pt')
                # Set device for Ultralytics models
                if self.device != 'cpu':
                    model.to(self.device)
                self.models[model_name] = model
            elif model_name == 'sam':
                model = SAM('../weights/sam2.1_b.pt')
                # Force SAM on CPU as requested (MPS incompatible)
                model.to('cpu')
                self.models[model_name] = model
            elif model_name == 'fastsam':
                model = FastSAM('../weights/FastSAM-s.pt')
                # Set device for Ultralytics models
                if self.device != 'cpu':
                    model.to(self.device)
                self.models[model_name] = model
            else:
                print(f"❌ Unknown model: {model_name}")
                return False
            
            print(f"✅ {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"🗑️  {model_name} unloaded from memory")
    
    def _load_u2net(self) -> torch.nn.Module:
        """Load U2Net model"""
        model = u2net_module.U2NET(3, 1)
        if hasattr(config, 'U2NET_WEIGHTS_PATH'):
            model.load_state_dict(torch.load(config.U2NET_WEIGHTS_PATH, map_location=self.device))
        else:
            # Try common paths
            weight_paths = [
                'weights/u2net.pth',
                'models/u2net.pth',
                'checkpoints/u2net.pth'
            ]
            for path in weight_paths:
                if os.path.exists(path):
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    break
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_hrnet(self) -> torch.nn.Module:
        """Load HRNet model"""
        model = hrnet_module.HighResolutionNet()
        if hasattr(config, 'HRNET_WEIGHTS_PATH'):
            model.load_state_dict(torch.load(config.HRNET_WEIGHTS_PATH, map_location=self.device))
        else:
            # Try common paths
            weight_paths = [
                'weights/hrnet.pth',
                'models/hrnet.pth', 
                'checkpoints/hrnet.pth'
            ]
            for path in weight_paths:
                if os.path.exists(path):
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    break
        
        model.to(self.device)
        model.eval()
        return model
    
    def segment_frame_u2net(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment frame using U2Net"""
        start_time = time.time()
        # Preprocess to match temp/silhouette.py
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(frame_rgb).unsqueeze(0).to(self.device)

        # Inference (U2Net returns multiple outputs)
        with torch.no_grad():
            outputs = self.models['u2net'](input_tensor)
            # Use primary output d0
            d0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred = d0[:, 0, :, :]
            pred = torch.sigmoid(pred)
            # Min-max normalize to 0-1 then to numpy
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred_np = pred.squeeze().detach().cpu().numpy()

        # Resize back to original
        pred_np = cv2.resize(pred_np, (frame.shape[1], frame.shape[0]))
        mask = (pred_np > 0.5).astype(np.uint8) * 255

        inference_time = time.time() - start_time
        return mask, inference_time
    
    def segment_frame_hrnet(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment frame using HRNet"""
        start_time = time.time()
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (512, 512))
        frame_tensor = torch.from_numpy(frame_resized).float()
        frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.models['hrnet'](frame_tensor)
            pred = torch.softmax(pred, dim=1)
            # Assuming person class is at index 15 (COCO person class)
            # For MPS compatibility, move to CPU before numpy conversion
            person_mask = pred[0, 15].cpu().numpy()
        
        # Post-process
        person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))
        mask = (person_mask > 0.5).astype(np.uint8) * 255
        
        inference_time = time.time() - start_time
        return mask, inference_time
    
    def segment_frame_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment frame using YOLO11n-seg"""
        start_time = time.time()
        
        # Inference - Ultralytics handles device automatically
        results = self.models['yolo11n_seg'](frame, verbose=False, device=self.device)
        
        # Extract person masks
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        # Also collect person boxes to guide SAM/FastSAM later
        collected_boxes: List[List[float]] = []
        
        for result in results:
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks
                num = len(masks.data)
                for i in range(num):
                    cls_id = int(boxes.cls[i].item()) if hasattr(boxes, 'cls') else int(boxes[i].cls)
                    if cls_id == 0:  # person
                        seg_mask = masks.data[i].detach().cpu().numpy()
                        seg_mask_resized = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                        mask = np.maximum(mask, (seg_mask_resized > 0.5).astype(np.uint8) * 255)
                        # xyxy box
                        if hasattr(boxes, 'xyxy'):
                            xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
                        else:
                            # Fallback
                            xyxy = [float(x) for x in boxes[i].xyxy[0].tolist()]
                        collected_boxes.append(xyxy)
        
        inference_time = time.time() - start_time
        # Save for later use (process_video will distribute to per-frame store)
        self._last_yolo_bboxes = collected_boxes
        return mask, inference_time
    
    def segment_frame_sam(self, frame: np.ndarray, bboxes: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, float]:
        """Segment frame using SAM (Ultralytics)"""
        start_time = time.time()
        
        # For SAM, run on CPU and prompt with person boxes when available
        sam_device = 'cpu'
        if bboxes and len(bboxes) > 0:
            try:
                results = self.models['sam'](frame, verbose=False, device=sam_device, bboxes=bboxes)
            except Exception:
                results = self.models['sam'](frame, verbose=False, device=sam_device)
        else:
            results = self.models['sam'](frame, verbose=False, device=sam_device)
        
        # Combine all masks that might contain persons
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks_data = result.masks.data.detach().cpu().numpy()
                if len(masks_data) > 0:
                    if bboxes and len(bboxes) > 0:
                        # Combine masks within prompted regions
                        combined = np.zeros_like(masks_data[0], dtype=np.float32)
                        for m in masks_data:
                            combined = np.maximum(combined, m.astype(np.float32))
                        target = combined
                    else:
                        # Fallback: largest mask
                        areas = [np.sum(m) for m in masks_data]
                        target = masks_data[int(np.argmax(areas))]
                    if target.shape != (frame.shape[0], frame.shape[1]):
                        target = cv2.resize(target.astype(np.float32), (frame.shape[1], frame.shape[0]))
                    mask = np.maximum(mask, (target > 0.5).astype(np.uint8) * 255)
        
        inference_time = time.time() - start_time
        return mask, inference_time
    
    def segment_frame_fastsam(self, frame: np.ndarray, bboxes: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, float]:
        """Segment frame using FastSAM (Ultralytics)"""
        start_time = time.time()
        
        # FastSAM inference - try prompting with boxes if available
        try:
            results = self.models['fastsam'](frame, verbose=False, device=self.device,
                                             bboxes=bboxes if bboxes and len(bboxes) > 0 else None)
        except Exception:
            results = self.models['fastsam'](frame, verbose=False, device=self.device)
        
        # Combine relevant masks
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks_data = result.masks.data.detach().cpu().numpy()
                if len(masks_data) > 0:
                    # Combine all masks (FastSAM generates multiple segments)
                    combined_mask = np.zeros_like(masks_data[0], dtype=np.float32)
                    for mask_data in masks_data:
                        combined_mask = np.maximum(combined_mask, mask_data.astype(np.float32))
                    
                    # Resize to frame size if needed
                    if combined_mask.shape != (frame.shape[0], frame.shape[1]):
                        combined_mask = cv2.resize(combined_mask.astype(np.float32),
                                                 (frame.shape[1], frame.shape[0]))
                    
                    mask = (combined_mask > 0.5).astype(np.uint8) * 255
        
        inference_time = time.time() - start_time
        return mask, inference_time
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def calculate_metrics(self, pred_mask: np.ndarray, gt_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate segmentation quality metrics"""
        metrics = {}
        
        # Basic mask statistics
        metrics['mask_area'] = np.sum(pred_mask > 0)
        metrics['mask_coverage'] = metrics['mask_area'] / (pred_mask.shape[0] * pred_mask.shape[1])
        
        # If ground truth is available, calculate IoU and Dice
        if gt_mask is not None:
            pred_binary = (pred_mask > 127).astype(np.uint8)
            gt_binary = (gt_mask > 127).astype(np.uint8)
            
            intersection = np.sum(pred_binary & gt_binary)
            union = np.sum(pred_binary | gt_binary)
            
            if union > 0:
                metrics['iou'] = intersection / union
                metrics['dice'] = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary))
            else:
                metrics['iou'] = 0.0
                metrics['dice'] = 0.0
        
        return metrics
    
    def process_video(self, video_path: str, max_frames: int = 50, 
                     output_dir: str = 'comparison_results') -> Dict[str, Any]:
        """Process video and compare all models sequentially"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Open video and read all frames first
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎬 Processing video: {video_path}")
        print(f"📊 Total frames: {total_frames}, FPS: {fps}, Processing: {max_frames} frames")
        
        # Read frames into memory first
        frames = []
        frame_count = 0
        print("📥 Reading frames into memory...")
        
        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
            print(f"\r🔄 Reading frame {frame_count}/{max_frames}", end='', flush=True)
        
        cap.release()
        print(f"\n✅ Read {len(frames)} frames into memory")
        
    # Define models to test (run YOLO first to collect person masks)
        model_names = ['yolo11n_seg', 'u2net', 'hrnet', 'sam', 'fastsam']
        results = {model_name: {
            'inference_times': [],
            'memory_usage': [],
            'metrics': [],
            'fps': []
        } for model_name in model_names}
        
    # Prepare person bbox store for SAM/FastSAM prompting and mask store
        self.person_bboxes_per_frame = [[] for _ in range(len(frames))]
        self.person_masks_per_frame = [None for _ in range(len(frames))]

        # Process each model sequentially
        for model_idx, model_name in enumerate(model_names):
            print(f"\n🚀 Processing with {model_name} ({model_idx + 1}/{len(model_names)})")
            
            # Load current model
            if not self.load_model(model_name):
                print(f"⚠️  Skipping {model_name} due to loading error")
                continue
            
            # Process all frames with this model
            for frame_idx, frame in enumerate(frames):
                try:
                    print(f"\r🔄 {model_name}: Processing frame {frame_idx + 1}/{len(frames)}", end='', flush=True)
                    
                    # Measure initial memory
                    initial_memory = self.get_memory_usage()
                    
                    # Segment frame
                    if model_name == 'u2net':
                        mask, inference_time = self.segment_frame_u2net(frame)
                        # Restrict to person region if YOLO mask available
                        if self.person_masks_per_frame and self.person_masks_per_frame[frame_idx] is not None:
                            mask = cv2.bitwise_and(mask, self.person_masks_per_frame[frame_idx])
                    elif model_name == 'hrnet':
                        mask, inference_time = self.segment_frame_hrnet(frame)
                        if self.person_masks_per_frame and self.person_masks_per_frame[frame_idx] is not None:
                            mask = cv2.bitwise_and(mask, self.person_masks_per_frame[frame_idx])
                    elif model_name == 'yolo11n_seg':
                        mask, inference_time = self.segment_frame_yolo(frame)
                        # Capture person boxes for this frame to guide SAM/FastSAM later
                        if hasattr(self, '_last_yolo_bboxes'):
                            self.person_bboxes_per_frame[frame_idx] = self._last_yolo_bboxes
                        # Save unified person mask for restricting other models
                        self.person_masks_per_frame[frame_idx] = mask.copy()
                    elif model_name == 'sam':
                        boxes = self.person_bboxes_per_frame[frame_idx] if self.person_bboxes_per_frame else None
                        mask, inference_time = self.segment_frame_sam(frame, bboxes=boxes)
                        if self.person_masks_per_frame and self.person_masks_per_frame[frame_idx] is not None:
                            mask = cv2.bitwise_and(mask, self.person_masks_per_frame[frame_idx])
                    elif model_name == 'fastsam':
                        boxes = self.person_bboxes_per_frame[frame_idx] if self.person_bboxes_per_frame else None
                        mask, inference_time = self.segment_frame_fastsam(frame, bboxes=boxes)
                        if self.person_masks_per_frame and self.person_masks_per_frame[frame_idx] is not None:
                            mask = cv2.bitwise_and(mask, self.person_masks_per_frame[frame_idx])
                    else:
                        continue
                    
                    # Measure final memory
                    final_memory = self.get_memory_usage()
                    memory_used = max(0, final_memory - initial_memory)  # Avoid negative values
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(mask)
                    
                    # Store results
                    results[model_name]['inference_times'].append(inference_time)
                    results[model_name]['memory_usage'].append(memory_used)
                    results[model_name]['metrics'].append(metrics)
                    results[model_name]['fps'].append(1.0 / inference_time if inference_time > 0 else 0)
                    
                    # Save first few masks for visual comparison
                    if frame_idx < 3:
                        mask_path = os.path.join(output_dir, f'{model_name}_frame_{frame_idx}_mask.jpg')
                        cv2.imwrite(mask_path, mask)
                
                except Exception as e:
                    print(f"\n❌ Error processing frame {frame_idx} with {model_name}: {e}")
                    continue
            
            print(f"\n✅ Completed {model_name} processing")
            
            # Print intermediate results
            if results[model_name]['inference_times']:
                avg_time = np.mean(results[model_name]['inference_times'])
                avg_fps = np.mean(results[model_name]['fps'])
                print(f"   📈 {model_name}: {avg_fps:.2f} FPS, {avg_time*1000:.2f}ms avg")
            
            # Unload current model to free memory
            self.unload_model(model_name)
            
            # Force garbage collection between models
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"\n🎉 All models processed!")
        
        # Generate summary statistics
        summary = self.generate_summary(results)
        
        # Save results
        self.save_results(results, summary, output_dir)
        
        return {'results': results, 'summary': summary, 'processed_frames': len(frames)}
    
    def generate_summary(self, results: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics"""
        summary = {}
        
        for model_name, model_results in results.items():
            if not model_results['inference_times']:
                continue
                
            summary[model_name] = {
                'avg_inference_time': np.mean(model_results['inference_times']),
                'std_inference_time': np.std(model_results['inference_times']),
                'avg_fps': np.mean(model_results['fps']),
                'max_fps': np.max(model_results['fps']),
                'avg_memory_usage': np.mean(model_results['memory_usage']),
                'max_memory_usage': np.max(model_results['memory_usage']),
                'avg_mask_coverage': np.mean([m['mask_coverage'] for m in model_results['metrics']])
            }
        
        return summary
    
    def save_results(self, results: Dict, summary: Dict, output_dir: str) -> None:
        """Save comparison results"""
        
        # Save detailed results as JSON
        import json
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'inference_times': [float(x) for x in model_results['inference_times']],
                'memory_usage': [float(x) for x in model_results['memory_usage']],
                'fps': [float(x) for x in model_results['fps']],
                'metrics': [{k: float(v) for k, v in m.items()} for m in model_results['metrics']]
            }
        
        with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate and save plots
        self.generate_plots(summary, output_dir)
        
        # Print summary to console
        self.print_summary(summary)
    
    def generate_plots(self, summary: Dict, output_dir: str) -> None:
        """Generate comparison plots"""
        
        if not summary:
            return
        
        model_names = list(summary.keys())
        
        # 1. FPS Comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        fps_values = [summary[name]['avg_fps'] for name in model_names]
        plt.bar(model_names, fps_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
        plt.title('Average FPS Comparison')
        plt.ylabel('FPS')
        plt.xticks(rotation=45)
        
        # 2. Inference Time Comparison  
        plt.subplot(2, 2, 2)
        time_values = [summary[name]['avg_inference_time'] * 1000 for name in model_names]  # Convert to ms
        plt.bar(model_names, time_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
        plt.title('Average Inference Time')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        # 3. Memory Usage Comparison
        plt.subplot(2, 2, 3)
        memory_values = [summary[name]['avg_memory_usage'] for name in model_names]
        plt.bar(model_names, memory_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
        plt.title('Average Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        
        # 4. Mask Coverage Comparison
        plt.subplot(2, 2, 4)
        coverage_values = [summary[name]['avg_mask_coverage'] * 100 for name in model_names]  # Convert to percentage
        plt.bar(model_names, coverage_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
        plt.title('Average Mask Coverage')
        plt.ylabel('Coverage (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to {output_dir}/model_comparison.png")
    
    def print_summary(self, summary: Dict) -> None:
        """Print summary to console"""
        print("\n" + "="*80)
        print("🏆 SEGMENTATION MODEL COMPARISON SUMMARY")
        print("="*80)
        
        if not summary:
            print("❌ No results to display")
            return
        
        # Find best performing models
        best_fps = max(summary.keys(), key=lambda x: summary[x]['avg_fps'])
        best_speed = min(summary.keys(), key=lambda x: summary[x]['avg_inference_time'])
        best_memory = min(summary.keys(), key=lambda x: summary[x]['avg_memory_usage'])
        
        print(f"🚀 Fastest FPS: {best_fps} ({summary[best_fps]['avg_fps']:.2f} FPS)")
        print(f"⚡ Lowest Latency: {best_speed} ({summary[best_speed]['avg_inference_time']*1000:.2f} ms)")
        print(f"🧠 Most Memory Efficient: {best_memory} ({summary[best_memory]['avg_memory_usage']:.2f} MB)")
        print()
        
        # Detailed results table
        print(f"{'Model':<15} {'FPS':<8} {'Time(ms)':<10} {'Memory(MB)':<12} {'Coverage(%)':<12}")
        print("-" * 65)
        
        for model_name, stats in summary.items():
            print(f"{model_name:<15} {stats['avg_fps']:<8.2f} "
                  f"{stats['avg_inference_time']*1000:<10.2f} "
                  f"{stats['avg_memory_usage']:<12.2f} "
                  f"{stats['avg_mask_coverage']*100:<12.2f}")
        
        print("="*80)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Compare segmentation models for person silhouette generation')
    parser.add_argument('--video', type=str, default=config.INPUT_VIDEO, help='Path to input video file')
    parser.add_argument('--max_frames', type=int, default=50, help='Maximum number of frames to process')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='mps', choices=['auto', 'cuda', 'mps', 'cpu'], 
                        help='Device to run models on')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    # Initialize comparison framework
    comparison = SegmentationModelComparison(device=config.DEVICE)
    
    print("🎯 Sequential model comparison mode - models will be loaded one at a time")
    
    # Process video and compare models
    try:
        results = comparison.process_video(
            video_path=args.video,
            max_frames=args.max_frames,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Comparison completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

