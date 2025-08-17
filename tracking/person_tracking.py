from pathlib import Path
import numpy as np
import torch
from boxmot import BoostTrack, ByteTrack
from utils.data_manager import DataBuffer
import config


class PersonTracker:
    """
    Person tracker using BoxMOT built-in trackers.

    - Prefers BoostTrack (motion + appearance) when OSNet ReID weights are available.
    - Falls back to ByteTrack (motion-only) when ReID weights are missing.
    """

    def __init__(
        self,
        reid_weights_path: str | None = None,
        # Tuning knobs for BoostTrack (robust preset for crowded/similar uniforms)
        max_age: int = 90,
        min_hits: int = 3,
        det_thresh: float = 0.55,
        iou_threshold: float = 0.35,
        lambda_iou: float = 0.45,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.30,
        use_dlo_boost: bool = True,
        use_duo_boost: bool = True,
        dlo_boost_coef: float = 0.70,
        s_sim_corr: bool = True,
        use_rich_s: bool = True,
        use_sb: bool = True,
        use_vt: bool = True,
    ):
        # Select device: prioritize MPS (Apple), then CUDA, else CPU
        self.device = (
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

        weights_path = Path(reid_weights_path or config.REID_WEIGHTS)

        if not weights_path.exists():
            print(f"[BoxMOT] ReID weights not found at: {weights_path}. Falling back to ByteTrack (motion-only).")
            self.tracker = ByteTrack(track_thresh=det_thresh, track_buffer=int(max_age), match_thresh=0.8)
            return

        # Use appearance-aware tracker for higher ID stability
        print(f"[BoxMOT] Using BoostTrack with ReID weights at: {weights_path}")
        self.tracker = BoostTrack(
            reid_weights=weights_path,
            device=self.device,
            half=(self.device == 'cuda'),
            max_age=max_age,
            min_hits=min_hits,
            det_thresh=det_thresh,
            iou_threshold=iou_threshold,
            lambda_iou=lambda_iou,
            lambda_mhd=lambda_mhd,
            lambda_shape=lambda_shape,
            use_dlo_boost=use_dlo_boost,
            use_duo_boost=use_duo_boost,
            dlo_boost_coef=dlo_boost_coef,
            s_sim_corr=s_sim_corr,
            use_rich_s=use_rich_s,
            use_sb=use_sb,
            use_vt=use_vt,
        )

    def update_tracks(self, frame_id, buffer: DataBuffer, profiler):
        # Prepare detections in BoxMOT format: M x [x1, y1, x2, y2, conf, cls]
        detections = buffer.get_detections(frame_id)
        frame = buffer.get_frame(frame_id)

        dets_for_tracking = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            cls = 0  # person class
            dets_for_tracking.append([x1, y1, x2, y2, conf, cls])

        dets_array = (
            np.array(dets_for_tracking, dtype=np.float32)
            if dets_for_tracking else np.empty((0, 6), dtype=np.float32)
        )

        # Track update; BoxMOT returns M x [x1,y1,x2,y2,id,conf,cls,ind]
        profiler.start("BoxMOT_Update")
        results = self.tracker.update(dets_array, frame)
        profiler.stop("BoxMOT_Update")

        final_tracks = []
        for t in results:
            x1, y1, x2, y2, track_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            final_tracks.append([x1, y1, x2, y2, track_id])

        buffer.store_tracks(frame_id, final_tracks)