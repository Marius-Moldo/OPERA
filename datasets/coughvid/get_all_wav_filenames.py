import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # datasets/
from dataset_filename_utils import glob_to_npy

# Collect all segmented coughvid wav paths
glob_to_npy(
    "datasets/coughvid/wav_segmented",
    "*.wav",
    "datasets/coughvid/entire_wav_filenames_segmented.npy",
)
