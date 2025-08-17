"""
Central configuration for the Person Identification pipeline.

All paths, device selection, and tunable parameters live here.
Update values below instead of hardcoding in modules.
"""

from __future__ import annotations
import os

# Set MPS fallback BEFORE importing torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch

# -----------------------------------
# Device selection & MPS handling
# -----------------------------------
# PREFERRED_DEVICE can be: 'auto' | 'mps' | 'cuda' | 'cpu'
PREFERRED_DEVICE = 'auto'

def _select_device() -> str:
    """Select the best available device for computation"""
    if PREFERRED_DEVICE != 'auto':
        if PREFERRED_DEVICE == 'mps' and not torch.backends.mps.is_available():
            print(f"[WARNING] MPS requested but not available, falling back to CPU")
            return 'cpu'
        elif PREFERRED_DEVICE == 'cuda' and not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return PREFERRED_DEVICE
    
    # Auto-select best available device
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def setup_mps_fallback():
    """Setup MPS fallback for unsupported operations"""
    device = _select_device()
    if device == 'mps' and torch.backends.mps.is_available():
        # Fallback is already set at module level, just return status
        return True
    return False

DEVICE: str = _select_device()
MPS_FALLBACK_ENABLED = setup_mps_fallback()

print(f"[CONFIG] Selected device: {DEVICE}")
if MPS_FALLBACK_ENABLED:
    print("[CONFIG] MPS fallback enabled for unsupported operations")

# Headless Flag
HEADLESS = True  # Set to True for headless mode (no GUI), False for GUI display

# -----------------------------------
# Common paths
# -----------------------------------
ROOT_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(ROOT_DIR, '..', 'weights')

# Inputs/Outputs
INPUT_VIDEO = os.path.join(ROOT_DIR, '..', 'v_0', 'input', '3c.mp4')
OUTPUT_TRACKING_VIDEO = os.path.join(ROOT_DIR, 'tracking_output.mp4')
MAX_FRAMES = 800  # limit processing frames; set to -1 for full video

# Models / Weights
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, 'yolo11s.pt')
U2NET_WEIGHTS = os.path.join(WEIGHTS_DIR, 'u2net_human_seg.pth')
PARSING_WEIGHTS = os.path.join(WEIGHTS_DIR, 'human_parsing.pth')

# BoxMOT / BoostTrack ReID weights path (OSNet/LightMBN etc.)
REID_WEIGHTS = os.path.join(WEIGHTS_DIR, 'lmbn_n_cuhk03_d.pth')

# Demo/Test assets
SILH_TEST_IMAGE = os.path.join(ROOT_DIR, 'person_crop_2.jpg')

# -----------------------------------
# Tracker parameters
# -----------------------------------
# BoostTrack preset (robust for crowded scenes)
BT_MAX_AGE = 90
BT_MIN_HITS = 3
BT_DET_THRESH = 0.55
BT_IOU_THRESHOLD = 0.35
BT_LAMBDA_IOU = 0.45
BT_LAMBDA_MHD = 0.25
BT_LAMBDA_SHAPE = 0.30
BT_USE_DLO_BOOST = True
BT_USE_DUO_BOOST = True
BT_DLO_BOOST_COEF = 0.70
BT_S_SIM_CORR = True
BT_USE_RICH_S = True
BT_USE_SB = True
BT_USE_VT = True

# ByteTrack fallback params
BYTETRACK_MATCH_THRESH = 0.8

