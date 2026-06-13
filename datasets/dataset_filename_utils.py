"""Shared helpers for the dataset filename-list scripts (load-npy -> transform -> save-npy)."""
import glob
import numpy as np


def replace_in_filenames(input_path, old, new, output_path):
    """Load a filename array, replace `old` with `new` in each entry, save."""
    arr = np.load(input_path)
    np.save(output_path, np.array([f.replace(old, new) for f in arr]))


def glob_to_npy(folder, pattern, output_path):
    """Glob `folder/pattern` and save the matching paths as a .npy array."""
    np.save(output_path, glob.glob(folder + "/" + pattern))
