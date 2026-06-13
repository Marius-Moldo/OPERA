import numpy as np
import glob

folder = "datasets/coughvid/wav_segmented"

all_files = glob.glob(folder + "/*.wav")

np.save("datasets/coughvid/entire_wav_filenames_segmented.npy", all_files)
