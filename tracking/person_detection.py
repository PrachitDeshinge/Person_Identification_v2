import torch
import cv2
import numpy as np
from ultralytics import YOLO
from utils.data_manager import DataBuffer
import config

class PersonDetector:
    def __init__(self, model_path: str | None = None, batch_size: int = 4):
        self.device = config.DEVICE
        self.batch_size = batch_size
        weights = model_path or config.YOLO_WEIGHTS
        self.model = YOLO(weights, verbose=False)
        
        # Batch processing buffers
        self.frame_buffer = []
        self.frame_id_buffer = []
        
        # Optimize for MPS performance
        if self.device == 'mps':
            # Enable MPS optimizations
            torch.backends.mps.enabled = True
            # Warm up the model with a dummy inference
            dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
            with torch.no_grad():
                try:
                    _ = self.model(dummy_input, verbose=False)
                except:
                    pass
        
        print(f"[PersonDetector] Using device: {self.device}")
        print(f"[PersonDetector] Model warmed up for optimal MPS performance")
        print(f"[PersonDetector] Batch size: {self.batch_size}")
        if config.MPS_FALLBACK_ENABLED:
            print("[PersonDetector] MPS fallback enabled for unsupported operations")

    def detect_and_store(self, frame: 'np.ndarray', frame_id: int, buffer: DataBuffer):
        """
        Detects persons in a frame using batch processing for optimal performance.
        Stores the detection data, crops, and masks in the data buffer.
        """
        # Add frame to batch buffer
        self.frame_buffer.append(frame.copy())
        self.frame_id_buffer.append(frame_id)
        
        # Process batch when buffer is full or on forced flush
        if len(self.frame_buffer) >= self.batch_size:
            self._process_batch(buffer)
    
    def flush_batch(self, buffer: DataBuffer):
        """Force process any remaining frames in the batch buffer."""
        if self.frame_buffer:
            self._process_batch(buffer)
    
    def _process_batch(self, buffer: DataBuffer):
        """Process a batch of frames for optimal GPU utilization."""
        if not self.frame_buffer:
            return
        
        # Optimized batch inference call for MPS
        with torch.no_grad():  # Disable gradient computation for faster inference
            results = self.model(self.frame_buffer, verbose=False, device=self.device, 
                               conf=0.25, iou=0.4)  # Batch processing
        
        # Process each frame's results
        for i, (frame, frame_id, result) in enumerate(zip(self.frame_buffer, self.frame_id_buffer, results)):
            self._process_single_result(frame, frame_id, result, buffer)
        
        # Clear buffers
        self.frame_buffer.clear()
        self.frame_id_buffer.clear()
    
    def _process_single_result(self, frame: np.ndarray, frame_id: int, results, buffer: DataBuffer):
        """Process detection results for a single frame."""
        persons = []
        crops = []
        masks = []
        
        # Check if segmentation masks are available
        has_masks = hasattr(results, 'masks') and results.masks is not None
        
        for i, box in enumerate(results.boxes):
            # Filter for 'person' class (class_id=0)
            if int(box.cls.item()) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                
                # Extract person crop
                person_crop = frame[y1:y2, x1:x2].copy()
                
                # Extract mask if available
                person_mask = None
                if has_masks and i < len(results.masks.data):
                    # Get the segmentation mask for this detection
                    mask = results.masks.data[i].cpu().numpy()
                    # Resize mask to original frame size if needed
                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    person_mask = mask
                
                persons.append({
                    'bbox': np.array([x1, y1, x2, y2]), 
                    'conf': conf
                })
                crops.append(person_crop)
                masks.append(person_mask)
        
        # Store detection data, crops, and masks in the buffer
        buffer.store_detections(frame_id, persons)
        buffer.store_crops(frame_id, crops)
        if any(mask is not None for mask in masks):
            buffer.store_masks(frame_id, masks)
