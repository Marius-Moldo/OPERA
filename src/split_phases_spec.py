import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
sr = 16000

def extract_features(spectrogram, sr=None):
    """
    Extract relevant features for cough segmentation
    Assumes spectrogram shape is (time, frequency)
    """
    print(spectrogram.shape)
    features = {}

    # Energy profile across time (sum across frequency bins)
    energy = np.sum(spectrogram, axis=1)  # Changed from axis=0 to axis=1
    features['energy'] = energy

    # Spectral centroid (brightness indicator)
    if sr is not None:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        # Adjust freqs to match the number of frequency bins
        if len(freqs) != spectrogram.shape[1]:
            freqs = np.linspace(0, sr/2, spectrogram.shape[1])
        # Changed broadcasting to match (time, frequency) shape
        centroid = np.sum(spectrogram * freqs[np.newaxis, :], axis=1) / (np.sum(spectrogram, axis=1) + 1e-10)
        features['spectral_centroid'] = centroid
    else:
        # Approximate for image-based spectrograms
        freq_bins = np.arange(spectrogram.shape[1])  # Changed to shape[1] for frequency dimension
        centroid = np.sum(spectrogram * freq_bins[np.newaxis, :], axis=1) / (np.sum(spectrogram, axis=1) + 1e-10)
        features['spectral_centroid'] = centroid

    # Zero crossing rate (for detecting turbulence)
    energy_diff = np.diff(energy)
    zcr = np.sum(np.diff(np.sign(energy_diff)) != 0)
    features['zcr'] = zcr

    # Spectral rolloff (along frequency axis)
    cumsum = np.cumsum(spectrogram, axis=1)  # Changed from axis=0 to axis=1
    threshold = 0.85 * cumsum[:, -1]  # Get last column (max cumsum for each time frame)
    rolloff = np.argmax(cumsum >= threshold[:, np.newaxis], axis=1)
    features['spectral_rolloff'] = rolloff

    return features


def find_segmentation_point(features, spectrogram):
    """
    Find the transition point between explosive and intermediate+voiced phases
    """
    energy = features['energy']
    centroid = features['spectral_centroid']
    rolloff = features['spectral_rolloff']

    # Normalize features
    energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
    centroid_norm = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + 1e-10)

    # Find the peak of the explosive phase (usually the highest energy point in the first half)
    first_half = len(energy) // 2
    explosive_peak = np.argmax(energy[:first_half])

    # Look for the transition after the explosive peak
    # The transition is characterized by:
    # 1. Drop in energy
    # 2. Change in spectral characteristics
    # 3. More stable spectral content

    # Compute energy gradient
    energy_grad = np.gradient(energy_norm)

    # Look for significant drop after peak
    search_start = explosive_peak + 5  # Start searching a bit after the peak
    search_end = min(len(energy) - 10, explosive_peak + len(energy) // 3)

    if search_start >= search_end:
        search_start = explosive_peak
        search_end = len(energy) - 1

    # Combined score for transition detection
    transition_scores = []
    for i in range(search_start, search_end):
        # Energy drop score
        energy_drop = -energy_grad[i]

        # Spectral stability score (lower variance after transition)
        if i + 10 < len(centroid):
            stability_before = np.std(centroid[i - 10:i])
            stability_after = np.std(centroid[i:i + 10])
            stability_score = stability_before - stability_after
        else:
            stability_score = 0

        # Combined score
        score = energy_drop + 0.5 * stability_score
        transition_scores.append(score)

    if transition_scores:
        # Find the best transition point
        best_idx = np.argmax(transition_scores)
        segmentation_point = search_start + best_idx
    else:
        # Fallback: use a fixed proportion
        segmentation_point = int(len(energy) * 0.3)

    return segmentation_point


def visualize_segmentation(spectrogram, segmentation_point, original_path, output_dir, idx):
    """
    Create visualization with segmentation line
    Assumes spectrogram shape is (time, frequency)
    """
    plt.figure(figsize=(12, 6))

    # Display spectrogram - transpose for display so frequency is on y-axis
    if spectrogram.ndim == 2:
        plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Magnitude (dB)')

    # Add segmentation line
    plt.axvline(x=segmentation_point, color='red', linewidth=2, linestyle='--',
                label='Segmentation Point')

    # Add phase labels
    mid_explosive = segmentation_point // 2
    mid_voiced = (segmentation_point + spectrogram.shape[0]) // 2  # Changed to shape[0] for time dimension

    plt.text(mid_explosive, spectrogram.shape[1] * 0.9, 'Explosive Phase',  # shape[1] for frequency dimension
             color='white', fontsize=12, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.text(mid_voiced, spectrogram.shape[1] * 0.9, 'Intermediate + Voiced Phase',  # shape[1] for frequency dimension
             color='white', fontsize=12, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title(f'Cough Segmentation: {os.path.basename(original_path)}')
    plt.legend()

    # Save the figure
    output_path = os.path.join(output_dir, f'segmented_{idx + 1}_{os.path.basename(original_path)}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def process_spectrograms(output_dir, files, max_files=10):
    """
    Main function to process spectrograms
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    files = files[:max_files]

    results = []

    for idx, filepath in enumerate(files):
        print(f"\nProcessing {idx + 1}/{len(files)}: {filepath}")

        # Load spectrogram
        spectrogram = np.load(filepath)

        # Extract features
        features = extract_features(spectrogram, sr)

        # Find segmentation point
        seg_point = find_segmentation_point(features, spectrogram)

        # Visualize and save
        output_path = visualize_segmentation(spectrogram, seg_point,
                                             str(filepath), output_dir, idx)

        # Calculate phase durations (in percentage)
        total_frames = spectrogram.shape[0]  # Changed from shape[1] to shape[0] for time dimension
        explosive_percent = (seg_point / total_frames) * 100
        voiced_percent = 100 - explosive_percent

        results.append({
            'file': filepath,
            'segmentation_frame': seg_point,
            'total_frames': total_frames,
            'explosive_phase_percent': explosive_percent,
            'voiced_phase_percent': voiced_percent,
            'output_path': output_path
        })

        print(f"  - Segmentation at frame {seg_point}/{total_frames}")
        print(f"  - Explosive phase: {explosive_percent:.1f}%")
        print(f"  - Intermediate+Voiced phase: {voiced_percent:.1f}%")

    # Summary
    print("\n" + "=" * 50)
    print("SEGMENTATION SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Explosive phase: {result['explosive_phase_percent']:.1f}%")
        print(f"  Intermediate+Voiced: {result['voiced_phase_percent']:.1f}%")
        print(f"  Output: {result['output_path']}")

    return results


# Example usage
if __name__ == "__main__":
    # Set your input and output directories
    filenames = np.load(os.path.join("datasets", "covidUK", "entire_cough_filenames.npy"))
    filenames = [os.path.join( f"{f}.npy") for f in filenames]

    output_directory = os.path.join("datasets", "covidUK", "segmented")

    # Process the spectrograms
    results = process_spectrograms(output_directory, filenames, max_files=10)

    print(f"\nProcessing complete! Check the '{output_directory}' folder for results.")