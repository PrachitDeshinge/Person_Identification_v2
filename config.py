# Configuration for the Person Identification Pipeline

# Re-ID Manager Settings
REID_SIMILARITY_THRESHOLD = 0.7
REID_LOST_TRACK_BUFFER = 60
REID_FEATURE_UPDATE_ALPHA = 0.9
REID_SWAP_CONFIDENCE_MARGIN = 0.2

# BoxMOT / BoostTrack ReID weights path (set to your downloaded LightMBN or OSNet weights)
# Examples:
#   '../weights/lmbn_n_msmt17.pt'
#   '../weights/osnet_x1_0_msmt17.pt'
#   '../weights/osnet_ain_x1_0_msmt17_256x128.pth'
REID_WEIGHTS = '../weights/lmbn_n_cuhk03_d.pth'
