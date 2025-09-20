import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
import tqdm


def save_waveform_with_split_line(audio, sr, split_idx, output_path):
    duration = len(audio) / sr
    time = np.linspace(0, duration, len(audio))

    plt.figure(figsize=(12, 4))
    plt.plot(time, audio, label="Waveform")
    plt.axvline(x=split_idx / sr, color="r", linestyle="--", label="Split Point")
    plt.title("Cough Waveform with Split Line")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def detect_cough_peak(audio, sr, search_frac=0.6):
    # Convert to analytic signal for envelope
    envelope = np.abs(hilbert(audio))
    # Smooth it to suppress noise
    smooth_env = gaussian_filter1d(envelope, sigma=sr * 0.01)

    # Only search in the first `search_frac` portion
    max_search_idx = int(len(audio) * search_frac)
    search_env = smooth_env[:max_search_idx]

    # Find peaks within that window
    peaks, properties = find_peaks(
        search_env, height=np.max(search_env) * 0.3, distance=sr * 0.1
    )

    if len(peaks) == 0:
        peak_idx = np.argmax(search_env)
    else:
        peak_idx = peaks[np.argmax(properties["peak_heights"])]

    # Ensure we don't pick a point too close to edges
    min_len = max(int(0.05 * sr), int(len(audio) * 0.1))
    return np.clip(peak_idx, min_len, len(audio) - min_len)


def split_cough(audio, sr, search_frac=0.6, offset_ms=25):
    # detect base peak index
    idx = detect_cough_peak(audio, sr, search_frac)

    # add 25 ms to the split point
    offset_samples = int((offset_ms / 1000) * sr)
    idx += offset_samples

    # clip again so we stay within audio bounds
    min_len = max(int(0.05 * sr), int(len(audio) * 0.1))
    idx = np.clip(idx, min_len, len(audio) - min_len)

    return audio[:idx], audio[idx:], idx


def process_wav(wav_path, out1=None, out2=None, out_png=None):
    sr, audio = wavfile.read(wav_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    (
        p1,
        p2,
        idx,
    ) = split_cough(audio, sr)

    if out1:
        wavfile.write(out1, sr, p1)
    if out2:
        wavfile.write(out2, sr, p2)
    if out_png:
        save_waveform_with_split_line(audio, sr, idx, out_png)

    return p1, p2, sr, idx / sr


if __name__ == "__main__":
    input_folder = "datasets/covidUK/audio_segmented"
    files = glob.glob(os.path.join(input_folder, "*.wav"))

    out1_dir = "datasets/covidUK/audio_segmented_phase_1"
    out2_dir = "datasets/covidUK/audio_segmented_phase_2"
    plot_dir = "datasets/covidUK/audio_split_plots"

    os.makedirs(out1_dir, exist_ok=True)
    os.makedirs(out2_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    for path in tqdm.tqdm(files):
        name = os.path.basename(path)
        out1 = os.path.join(out1_dir, name)
        out2 = os.path.join(out2_dir, name)
        plot = os.path.join(plot_dir, name.replace(".wav", ".png"))

        # ◀︎ SKIP if both output wavs already exist
        if os.path.exists(out1) and os.path.exists(out2):
            continue

        # now split and save
        p1, p2, sr, split_time = process_wav(
            path,
            out1=out1,
            out2=out2,
            # uncomment if you need plots:
            # out_png=plot,
        )
