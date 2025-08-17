import torch
import cv2
from ultralytics import YOLO
from utils.data_manager import DataBuffer
import config

class PersonDetector:
    def __init__(self, model_path: str | None = None):
        self.device = config.DEVICE
        weights = model_path or config.YOLO_WEIGHTS
        self.model = YOLO(weights, verbose=False)
        
        print(f"[PersonDetector] Using device: {self.device}")
        if config.MPS_FALLBACK_ENABLED:
            print("[PersonDetector] MPS fallback enabled for unsupported operations")

    def detect_and_store(self, frame: 'np.ndarray', frame_id: int, buffer: DataBuffer):
        """
        Detects persons in a frame and stores the detection data (not the frame)
        in the data buffer.
        """
        # Pass device explicitly to YOLO inference
        results = self.model(frame, verbose=False, device=self.device)[0]
        
        persons = []
        for box in results.boxes:
            # Filter for 'person' class (class_id=0)
            if int(box.cls.item()) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                persons.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        
        # Store only the lightweight detection data, not the full frame
        buffer.store_detections(frame_id, persons)
