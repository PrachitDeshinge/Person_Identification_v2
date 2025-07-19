from boxmot import ByteTrack
import torch
import numpy as np
from data_manager import DataBuffer

class PersonTracker:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)

    def update_tracks(self, frame_id, buffer: DataBuffer):
        detections = buffer.get_detections(frame_id)
        frame = buffer.get_frame(frame_id)

        dets_for_tracking = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_id = 0  # person class
            dets_for_tracking.append([x1, y1, x2, y2, conf, class_id])

        dets_array = np.array(dets_for_tracking, dtype=np.float32) if dets_for_tracking else np.empty((0, 6), dtype=np.float32)
        tracks = self.tracker.update(dets_array, frame)
        buffer.store_tracks(frame_id, tracks)