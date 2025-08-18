import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import config
from models import u2net as u2net_module

class U2NetSilhouetteGenerator:
    def __init__(self, weights_path: str | None = None, batch_size: int = 32, input_size: int = 128):
        self.device = config.DEVICE
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = self._load_u2net(weights_path)
        
        # Ultra-optimized transform with very small input size for speed
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size), interpolation=transforms.InterpolationMode.NEAREST),  # Faster interpolation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Pre-allocate tensors to reduce allocation overhead
        self._init_tensors()

    def _load_u2net(self, weights_path: str | None) -> torch.nn.Module:
        """Load U2Net model with optimization for inference"""
        model = u2net_module.U2NET(3, 1)
        
        path = weights_path or config.U2NET_WEIGHTS_PATH
        
        if not path:
            raise FileNotFoundError(f"U2Net weights not found at {path}")

        model.load_state_dict(torch.load(str(path), map_location=self.device))
        model.to(self.device)
        model.eval()
        
        # Optimize for inference
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        
        # Enable optimized attention if available
        if hasattr(torch.backends, 'mps') and self.device == 'mps':
            # Ensure MPS is optimized for inference
            torch.mps.empty_cache()
        
        print(f"[U2NetSilhouetteGenerator] U2Net model loaded from {path} on device: {self.device}")
        print(f"[U2NetSilhouetteGenerator] Using input size: {self.input_size}x{self.input_size}, batch size: {self.batch_size}")
        return model

    def _init_tensors(self):
        """Pre-allocate tensors to reduce memory allocation overhead"""
        self.batch_tensor = torch.zeros((self.batch_size, 3, self.input_size, self.input_size), 
                                       device=self.device, dtype=torch.float32)
        
        # Pre-allocate output tensor for batch processing
        self.output_tensor = torch.zeros((self.batch_size, self.input_size, self.input_size), 
                                        device=self.device, dtype=torch.float32)

    def generate_silhouette(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """
        Generates a combined silhouette mask for all tracked persons in a frame.
        Uses batch processing for better performance.
        """
        if not tracks:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        frame_height, frame_width = frame.shape[:2]
        final_silhouette = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Collect all valid person crops and their coordinates
        valid_crops = []
        valid_coords = []
        
        for t in tracks:
            if len(t) < 5:
                continue  # Skip malformed tracks
                
            x1, y1, x2, y2, _ = t[:5]  # Handle extra elements gracefully
            
            # Validate and clamp coordinates to frame boundaries
            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width))
            y2 = max(0, min(int(y2), frame_height))
            
            # Skip invalid bounding boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract person crop from the frame
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            valid_crops.append(person_crop)
            valid_coords.append((x1, y1, x2, y2))
        
        if not valid_crops:
            return final_silhouette
            
        # Process crops in batches for better performance
        for i in range(0, len(valid_crops), self.batch_size):
            batch_crops = valid_crops[i:i + self.batch_size]
            batch_coords = valid_coords[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensor = self._preprocess_crops_batch(batch_crops)
            
            if batch_tensor.size(0) == 0:
                continue
            
            # Run batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                d0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                preds = torch.sigmoid(d0[:, 0, :, :])
                
                # Normalize predictions
                for j in range(preds.size(0)):
                    pred = preds[j]
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    preds[j] = pred
            
            # Process each prediction in the batch
            for j, (pred, coords) in enumerate(zip(preds, batch_coords)):
                if j >= len(batch_crops):
                    break
                    
                x1, y1, x2, y2 = coords
                crop_height, crop_width = batch_crops[j].shape[:2]
                
                # Resize prediction back to original crop size
                pred_np = pred.cpu().numpy()
                person_mask = cv2.resize(pred_np, (crop_width, crop_height))
                
                # Apply threshold to get binary mask
                person_mask = (person_mask > 0.5).astype(np.uint8)
                
                # Place the person's silhouette back into the full frame
                final_silhouette[y1:y2, x1:x2] = np.maximum(
                    final_silhouette[y1:y2, x1:x2], 
                    person_mask
                )
        
        return final_silhouette * 255

    def _preprocess_crops_batch(self, crops: list) -> torch.Tensor:
        """Efficiently preprocess multiple crops for batch inference"""
        if not crops:
            return torch.empty(0, 3, self.input_size, self.input_size, device=self.device)
        
        batch_size = len(crops)
        
        # Use pre-allocated tensor if batch fits
        if batch_size <= self.batch_size:
            batch_tensor = self.batch_tensor[:batch_size]
        else:
            # Create new tensor for larger batches
            batch_tensor = torch.zeros((batch_size, 3, self.input_size, self.input_size), 
                                     device=self.device, dtype=torch.float32)
        
        # Process each crop
        for i, crop in enumerate(crops):
            # Convert crop to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms and add to batch
            crop_tensor = self.transform(crop_rgb)
            batch_tensor[i] = crop_tensor.to(self.device)
        
        return batch_tensor
