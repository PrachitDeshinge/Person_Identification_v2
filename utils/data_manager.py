from __future__ import annotations

class DataBuffer:
    """
    Manages transient data between pipeline stages, specifically detections and tracks.

    This buffer is designed to be memory-efficient by NOT storing full video frames.
    It holds detections and track data only for a limited time window, defined
    by `max_buffer_size`, to support tracking algorithms that may look back
    a few frames.
    """
    def __init__(self, max_buffer_size: int = 150):
        self.detections: dict[int, list] = {}
        self.tracks: dict[int, list] = {}
        self.crops: dict[int, list] = {}
        self.masks: dict[int, list] = {}
        # Keep a buffer to prevent indefinite memory growth. Increased buffer size
        # is acceptable now that full frames are not stored.
        self.max_buffer_size = max_buffer_size

    def store_detections(self, frame_id: int, detections: list):
        """Stores detections for a given frame ID."""
        self.detections[frame_id] = detections

    def get_detections(self, frame_id: int) -> list:
        """Retrieves detections for a given frame ID."""
        return self.detections.get(frame_id, [])

    def store_tracks(self, frame_id: int, tracks: list):
        """Stores tracking results for a given frame ID."""
        self.tracks[frame_id] = tracks

    def get_tracks(self, frame_id: int) -> list:
        """Retrieves tracking results for a given frame ID."""
        return self.tracks.get(frame_id, [])

    def store_crops(self, frame_id: int, crops: list):
        """Stores person crops for a given frame ID."""
        self.crops[frame_id] = crops

    def get_crops(self, frame_id: int) -> list:
        """Retrieves person crops for a given frame ID."""
        return self.crops.get(frame_id, [])

    def store_masks(self, frame_id: int, masks: list):
        """Stores segmentation masks for a given frame ID."""
        self.masks[frame_id] = masks

    def get_masks(self, frame_id: int) -> list:
        """Retrieves segmentation masks for a given frame ID."""
        return self.masks.get(frame_id, [])

    def cleanup(self, current_frame_id: int):
        """
        Removes old detection and track data to manage memory.

        This prevents the buffer from growing indefinitely over long video streams.
        """
        if current_frame_id <= self.max_buffer_size:
            return
        
        cutoff_id = current_frame_id - self.max_buffer_size
        
        # Clean up old keys from both detections and tracks dictionaries
        old_detection_keys = [k for k in self.detections if k < cutoff_id]
        for k in old_detection_keys:
            del self.detections[k]
            
        old_track_keys = [k for k in self.tracks if k < cutoff_id]
        for k in old_track_keys:
            # Tracks may not exist for every detection frame, so check first
            if k in self.tracks:
                del self.tracks[k]
        
        # Clean up old crops and masks
        old_crop_keys = [k for k in self.crops if k < cutoff_id]
        for k in old_crop_keys:
            del self.crops[k]
            
        old_mask_keys = [k for k in self.masks if k < cutoff_id]
        for k in old_mask_keys:
            del self.masks[k]