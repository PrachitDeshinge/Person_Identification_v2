"""
Central configuration for the Person Identification pipeline.

This file is organized with the most frequently changed settings at the top
and the core setup logic at the bottom.

Sections:
1. High-Level Controls & I/O: Your main knobs for running the pipeline.
2. Model & Asset Paths: Locations for all required model weights.
3. Advanced Tracker Tuning: Parameters for fine-tuning tracking behavior.
4. Core Setup & Device Logic: The underlying setup that rarely needs changes.
"""

from __future__ import annotations
import os
import torch

# =================================================================================
# --- 1. High-Level Controls & I/O (Frequently Modified) ---
# =================================================================================

# -- Execution Settings --
# PREFERRED_DEVICE can be: 'auto' | 'mps' | 'cuda' | 'cpu'
PREFERRED_DEVICE = 'auto'
HEADLESS = True  # Set to True for headless mode (no GUI), False for GUI display

# -- Input/Output Paths --
# It's recommended to use absolute paths for robustness, but relative paths work too.
ROOT_DIR = os.path.dirname(__file__)
INPUT_VIDEO = os.path.join(ROOT_DIR, '../../../../Downloads', 'temp4.mp4')
OUTPUT_TRACKING_VIDEO = os.path.join(ROOT_DIR, 'tracking_output.mp4') # Set to None to disable saving video
OUTPUT_SILHOUETTE_VIDEO = os.path.join(ROOT_DIR, 'silhouette_output.mp4') # Set to None to disable saving video

# -- Processing Limits --
MAX_FRAMES = 500  # Limit processing frames; set to -1 for the full video


# =================================================================================
# --- 2. Model & Asset Paths ---
# =================================================================================
WEIGHTS_DIR = os.path.join(ROOT_DIR, '..', 'weights')

# -- Core Model Weights --
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, '../v_2/yolo11s_cihp_optimized/weights/best.pt')

# -- Re-Identification Weights for BoostTrack --
# If this file doesn't exist, the tracker falls back to motion-only (ByteTrack).
REID_WEIGHTS = os.path.join(WEIGHTS_DIR, 'lmbn_n_cuhk03_d.pth')

# =================================================================================
# --- 3. Advanced Tracker Tuning ---
# =================================================================================

# -- BoostTrack Parameters (used when REID_WEIGHTS are found) --
# This is a robust preset for crowded scenes or people with similar clothing.
BT_MAX_AGE = 90          # Max frames to keep a track without a new detection.
BT_MIN_HITS = 3          # Min detections to start a new track.
BT_DET_THRESH = 0.55     # Detection confidence threshold.
BT_IOU_THRESHOLD = 0.35  # IoU threshold for matching tracks to detections.
# Feature weighting for the matching cascade
BT_LAMBDA_IOU = 0.45     # Weight for IoU similarity.
BT_LAMBDA_MHD = 0.25     # Weight for Mahalanobis distance (motion).
BT_LAMBDA_SHAPE = 0.30   # Weight for bbox aspect ratio similarity.
# Advanced Boosts for handling occlusion and state estimation
BT_USE_DLO_BOOST = True
BT_USE_DUO_BOOST = True
BT_DLO_BOOST_COEF = 0.70
BT_S_SIM_CORR = True
BT_USE_RICH_S = True
BT_USE_SB = True
BT_USE_VT = True

# -- ByteTrack Fallback Parameters (used when REID_WEIGHTS are NOT found) --
BYTETRACK_MATCH_THRESH = 0.8


# =================================================================================
# --- 4. Core Setup & Device Logic (Should not need frequent changes) ---
# =================================================================================

def _select_device() -> str:
    """Selects the best available compute device based on user preference and availability."""
    if PREFERRED_DEVICE != 'auto':
        if PREFERRED_DEVICE == 'mps':
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            if not torch.backends.mps.is_available():
                # Set PyTorch MPS fallback for Apple Silicon devices
                print("[WARNING] MPS requested but not available. Falling back to CPU.")
                return 'cpu'
            return 'mps'
        elif PREFERRED_DEVICE == 'cuda':
            if not torch.cuda.is_available():
                print("[WARNING] CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
            return 'cuda:0'  # Use explicit device 0 for compatibility
        return PREFERRED_DEVICE
    
    # Auto-select the best available device
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'

def setup_mps_fallback() -> bool:
    """Checks if the MPS fallback environment variable is active."""
    device = _select_device()
    if device == 'mps' and torch.backends.mps.is_available():
        return os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
    return False

# --- Final Device Initialization & Logging ---
DEVICE: str = _select_device()
MPS_FALLBACK_ENABLED = setup_mps_fallback()

print("-" * 50)
print(f"⚙️  Configuration Loaded:")
print(f"    - Compute Device: {DEVICE}")
if DEVICE == 'mps':
    print(f"    - MPS Fallback: {'Enabled' if MPS_FALLBACK_ENABLED else 'Disabled'}")
print(f"    - Headless Mode: {'On' if HEADLESS else 'Off'}")
print(f"    - Input Video: '{os.path.basename(INPUT_VIDEO)}'")
print("-" * 50)