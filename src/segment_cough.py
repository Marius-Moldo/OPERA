import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import sys
import glob
import soundfile as sf
from tqdm import tqdm
from torchvision.datasets import folder


# from https://github.com/bagustris/detect-segment-cough/blob/master/src/segmentation.py


# Use old segmentation
def segment_cough(
    x,
    fs,
    cough_padding=0.5,
    min_cough_len=0.001,
    th_l_multiplier=0.1,
    th_h_multiplier=2,
):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power

    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a low threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator

    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress
    """

    cough_mask = np.array([False] * len(x))

    # define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    # segment coughs
    coughSegments = []
    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            # counting and adding cough samples
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                        coughSegments.append(x[cough_start : cough_end + 1])
                        cough_mask[cough_start : cough_end + 1] = True
            # cough end
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                    coughSegments.append(x[cough_start : cough_end + 1])
            # reset counter for number of sample tolerance
            else:
                below_th_counter = 0
        else:
            # start cough
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask


def compute_SNR(x, fs):
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    segments, cough_mask = segment_cough(x, fs)
    RMS_signal = (
        0 if len(x[cough_mask]) == 0 else np.sqrt(np.mean(np.square(x[cough_mask])))
    )
    RMS_noise = np.sqrt(np.mean(np.square(x[~cough_mask])))
    SNR = (
        0
        if (RMS_signal == 0 or np.isnan(RMS_noise))
        else 20 * np.log10(RMS_signal / RMS_noise)
    )
    return SNR


def segment_and_plot_coughs(files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        print(f"Processing {file}")
        x, fs = librosa.load(file, sr=None)
        plt.plot(x)
        plt.show()

        # segments cough
        cough_segments, cough_mask = segment_cough(x, fs, cough_padding=0.0)
        plt.plot(x)
        plt.plot(cough_mask)
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(file)}.png"))
        plt.close()


def segment_and_save_coughs(files, output_dir=None):
    """
    Segments coughs from a list of audio files and saves each segment as a new WAV file.

    Args:
        files (list): A list of paths to the input audio files.
        output_dir (str): The directory where the segmented cough WAV files will be saved.
    """
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    for file_path in tqdm(files):
        try:
            # Load the audio file
            x, fs = librosa.load(file_path, sr=None)

            # Segment the audio to find coughs
            cough_segments, _ = segment_cough(x, fs, cough_padding=0.2)

            if not cough_segments:
                print(f"  -> No coughs detected in {os.path.basename(file_path)}")
                continue

            # Get the base name of the original file to create new filenames
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Save each cough segment to a new WAV file
            for i, segment in enumerate(cough_segments):
                # Create a unique filename for each segment
                output_filename = f"{base_name}_cough_{i+1}.wav"
                output_path = os.path.join(output_dir, output_filename)

                # Write the segment to a WAV file using the soundfile library
                sf.write(output_path, segment, fs)

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")


if __name__ == "__main__":
    files = glob.glob(os.path.join("datasets", "coughvid", "wav", "*.wav"))

    wav_output_dir = os.path.join("datasets", "coughvid", "wav_segmented")
    segment_and_save_coughs(files, wav_output_dir)
