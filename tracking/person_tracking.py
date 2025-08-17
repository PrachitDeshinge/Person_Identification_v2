from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from boxmot import BoostTrack, ByteTrack
from utils.data_manager import DataBuffer
from utils.profiler import PipelineProfiler
import config
from typing import Optional


class PersonTracker:
    """
    Person tracker using BoxMOT built-in trackers.

    - Prefers BoostTrack (motion + appearance) when OSNet ReID weights are available.
    - Falls back to ByteTrack (motion-only) when ReID weights are missing.
    """

    def __init__(
        self,
        reid_weights_path: Optional[str] = None,
        # Tuning knobs for BoostTrack (robust preset for crowded/similar uniforms)
        max_age: int = config.BT_MAX_AGE,
        min_hits: int = config.BT_MIN_HITS,
        det_thresh: float = config.BT_DET_THRESH,
        iou_threshold: float = config.BT_IOU_THRESHOLD,
        lambda_iou: float = config.BT_LAMBDA_IOU,
        lambda_mhd: float = config.BT_LAMBDA_MHD,
        lambda_shape: float = config.BT_LAMBDA_SHAPE,
        use_dlo_boost: bool = config.BT_USE_DLO_BOOST,
        use_duo_boost: bool = config.BT_USE_DUO_BOOST,
        dlo_boost_coef: float = config.BT_DLO_BOOST_COEF,
        s_sim_corr: bool = config.BT_S_SIM_CORR,
        use_rich_s: bool = config.BT_USE_RICH_S,
        use_sb: bool = config.BT_USE_SB,
        use_vt: bool = config.BT_USE_VT,
    ):
        # Select device from central config
        self.device = config.DEVICE

        weights_path = Path(reid_weights_path or config.REID_WEIGHTS)

        if not weights_path.exists():
            print(f"[BoxMOT] ReID weights not found at: {weights_path}. Falling back to ByteTrack (motion-only).")
            self.tracker = ByteTrack(track_thresh=det_thresh, track_buffer=int(max_age), match_thresh=config.BYTETRACK_MATCH_THRESH)
            return

        # Use appearance-aware tracker for higher ID stability
        print(f"[BoxMOT] Using BoostTrack with ReID weights at: {weights_path}")
        self.tracker = BoostTrack(
            reid_weights=weights_path,
            device=self.device,
            half=(self.device == 'cuda'),
            # max_age=max_age,
            # min_hits=min_hits,
            # det_thresh=det_thresh,
            # iou_threshold=iou_threshold,
            # lambda_iou=lambda_iou,
            # lambda_mhd=lambda_mhd,
            # lambda_shape=lambda_shape,
            # use_dlo_boost=use_dlo_boost,
            # use_duo_boost=use_duo_boost,
            # dlo_boost_coef=dlo_boost_coef,
            # s_sim_corr=s_sim_corr,
            # use_rich_s=use_rich_s,
            # use_sb=use_sb,
            # use_vt=use_vt,
        )

    def update_tracks(self, frame: np.ndarray, frame_id: int, buffer: DataBuffer, profiler: PipelineProfiler):
        """
        Updates tracks with new detections for the current frame.
        The full frame is required by BoostTrack to extract appearance embeddings.
        """
        detections = buffer.get_detections(frame_id)

        if not detections:
            # If no detections, still update tracker to process existing tracks (e.g., for aging)
            dets_array = np.empty((0, 6), dtype=np.float32)
        else:
            # Prepare detections in BoxMOT format: [x1, y1, x2, y2, conf, cls]
            dets_for_tracking = [
                det['bbox'] + [det['conf'], 0] for det in detections  # 0 is person class
            ]
            dets_array = np.array(dets_for_tracking, dtype=np.float32)

        # Track update; BoxMOT returns M x [x1,y1,x2,y2,id,conf,cls,ind]
        profiler.start("BoxMOT_Update")
        # The `frame` is crucial here for the ReID model in BoostTrack
        results = self.tracker.update(dets_array, frame)
        profiler.stop("BoxMOT_Update")

        final_tracks = []
        if results.size > 0:
            # Extract [x1, y1, x2, y2, track_id]
            final_tracks = [list(map(int, t[:5])) for t in results]

        buffer.store_tracks(frame_id, final_tracks)