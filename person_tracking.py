from boxmot import ByteTrack
import torch
import numpy as np
from data_manager import DataBuffer
from reid_manager import ReIDManager

class PersonTracker:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tracker = ByteTrack(track_thresh=0.5, track_buffer=60, match_thresh=0.8)
        self.reid_manager = ReIDManager()
        self.active_track_ids = set()
        self.track_id_map = {}
        self.next_final_id = 1

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
                
                profiler.start("ReID_Extract_Feature")
                feature = self.reid_manager.extract_feature(crop)
                profiler.stop("ReID_Extract_Feature")

                profiler.start("ReID_Match_Feature")
                matched_id = self.reid_manager.match_feature(feature)
                profiler.stop("ReID_Match_Feature")

                if matched_id is not None:
                    # This is a re-identified track.
                    final_id = matched_id
                    # print(f"Re-identified raw track {raw_id} as existing ID {final_id}")
                else:
                    # This is a truly new track.
                    final_id = self.next_final_id
                    self.next_final_id += 1
                
                self.track_id_map[raw_id] = final_id
                self.reid_manager.register_feature(final_id, feature)

            else:
                # This raw ID has been seen before. Use its mapped final ID.
                final_id = self.track_id_map[raw_id]
            
            final_tracks.append([x1, y1, x2, y2, final_id])
            final_current_track_ids.add(final_id)

        # Identify which FINAL IDs are no longer present.
        profiler.start("ReID_Handle_Lost_Tracks")
        lost_final_ids = self.active_track_ids - final_current_track_ids
        self.reid_manager.handle_lost_tracks(lost_final_ids, frame_id)
        profiler.stop("ReID_Handle_Lost_Tracks")

        profiler.start("ReID_Cleanup_Lost_Gallery")
        self.reid_manager.cleanup_lost_gallery(frame_id)
        profiler.stop("ReID_Cleanup_Lost_Gallery")

        # Clean up the map from raw IDs that are no longer tracked by ByteTrack.
        raw_lost_ids = set(self.track_id_map.keys()) - raw_current_track_ids
        for raw_id in raw_lost_ids:
            del self.track_id_map[raw_id]

        # Update the set of active FINAL IDs for the next frame.
        self.active_track_ids = final_current_track_ids
        buffer.store_tracks(frame_id, final_tracks)