import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # datasets/
from dataset_filename_utils import replace_in_filenames

# Point the coughvid spectrogram filenames at the segmented wav files
replace_in_filenames(
    "datasets/coughvid/entire_spec_filenames.npy",
    "entire_spec_npy",
    "wav_segmented",
    "datasets/coughvid/entire_cough_filenames_segmented.npy",
)
