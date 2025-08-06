from boxmot import ByteTrack
import torch
import numpy as np
from utils.data_manager import DataBuffer
from models.reid_manager import ReIDManager
from models.u2net import U2NET
from torchvision import transforms
from PIL import Image
import cv2

RECHECK_INTERVAL = 10  # frames

class PersonTracker:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tracker = ByteTrack(track_thresh=0.5, track_buffer=60, match_thresh=0.8)
        self.reid_manager = ReIDManager()

        # Segmentation model setup
        self.segmentation_model = U2NET(3, 1)
        try:
            self.segmentation_model.load_state_dict(torch.load('../weights/u2net.pth', map_location=self.device))
            print("[SUCCESS] Segmentation model loaded successfully!")
        except FileNotFoundError:
            print("[ERROR] Segmentation model weights not found at '../weights/u2net.pth'. Proceeding without segmentation.")
            self.segmentation_model = None
        
        if self.segmentation_model:
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()

        self.segmentation_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.active_track_ids = set()
        self.track_id_map = {}
        self.next_final_id = 1
        self.recheck_counter = {}

    def segment_and_apply_mask(self, crop):
        if not self.segmentation_model:
            return crop

        original_h, original_w, _ = crop.shape
        image = Image.fromarray(crop).convert("RGB")
        image_tensor = self.segmentation_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            d0, _, _, _, _, _, _ = self.segmentation_model(image_tensor)

        pred = d0[:, 0, :, :]
        pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred)) # Normalize
        pred = pred.squeeze().cpu().numpy()
        
        mask = (pred > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask_resized)
        return masked_crop

    def update_tracks(self, frame_id, buffer: DataBuffer, profiler):
        detections = buffer.get_detections(frame_id)
        frame = buffer.get_frame(frame_id)

        dets_for_tracking = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_id = 0  # person class
            dets_for_tracking.append([x1, y1, x2, y2, conf, class_id])

        dets_array = np.array(dets_for_tracking, dtype=np.float32) if dets_for_tracking else np.empty((0, 6), dtype=np.float32)
        
        profiler.start("ByteTrack_Update")
        raw_tracks = self.tracker.update(dets_array, frame)
        profiler.stop("ByteTrack_Update")

        final_tracks = []
        final_current_track_ids = set()
        raw_current_track_ids = set()

        for t in raw_tracks:
            x1, y1, x2, y2, raw_id = map(int, t[:5])
            raw_current_track_ids.add(raw_id)

            if raw_id not in self.track_id_map:
                # New raw ID from ByteTrack.
                crop = frame[y1:y2, x1:x2]
                masked_crop = self.segment_and_apply_mask(crop)
                feature = self.reid_manager.extract_feature(masked_crop)
                matched_id = self.reid_manager.match_feature(feature)

                if matched_id is not None:
                    final_id = matched_id
                else:
                    final_id = self.next_final_id
                    self.next_final_id += 1
                
                self.track_id_map[raw_id] = final_id
                self.reid_manager.register_feature(final_id, feature)
                self.recheck_counter[final_id] = 0
            else:
                final_id = self.track_id_map[raw_id]
                self.recheck_counter[final_id] = self.recheck_counter.get(final_id, 0) + 1

                if self.recheck_counter[final_id] % RECHECK_INTERVAL == 0:
                    crop = frame[y1:y2, x1:x2]
                    masked_crop = self.segment_and_apply_mask(crop)
                    feature = self.reid_manager.extract_feature(masked_crop)
                    
                    corrected_id = self.reid_manager.check_for_swap(feature, final_id)
                    if corrected_id != final_id:
                        # A high-confidence swap was detected. Correct the mapping.
                        self.track_id_map[raw_id] = corrected_id
                        final_id = corrected_id
                    
                    # Update the feature vector for the correct ID
                    self.reid_manager.update_feature(final_id, feature)

            final_tracks.append([x1, y1, x2, y2, final_id])
            final_current_track_ids.add(final_id)

        # --- Track Lifecycle Management ---
        lost_final_ids = self.active_track_ids - final_current_track_ids
        self.reid_manager.handle_lost_tracks(lost_final_ids, frame_id)
        for final_id in lost_final_ids:
            if final_id in self.recheck_counter:
                del self.recheck_counter[final_id]

        self.reid_manager.cleanup_lost_gallery(frame_id)

        raw_lost_ids = set(self.track_id_map.keys()) - raw_current_track_ids
        for raw_id in raw_lost_ids:
            del self.track_id_map[raw_id]

        self.active_track_ids = final_current_track_ids
        buffer.store_tracks(frame_id, final_tracks)