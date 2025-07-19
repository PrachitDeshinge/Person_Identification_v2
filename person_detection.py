import torch
import cv2
from ultralytics import YOLO
from data_manager import DataBuffer

class PersonDetector:
    def __init__(self, model_path='../weights/yolov8n.pt'):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path,verbose=False).to(self.device)

    def detect_and_store(self, frame, frame_id, buffer: DataBuffer):
        results = self.model(frame,verbose=False)[0]
        persons = []
        for box in results.boxes:
            cls = int(box.cls.item())
            if cls == 0:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                persons.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        buffer.store_detections(frame_id, frame, persons)
