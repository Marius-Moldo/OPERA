import glob
import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.linalg import svd
import librosa
import soundfile as sf
from autrainer.datasets import BaseClassificationDataset
from tqdm import tqdm


def apply_svd_smoothing_audio(
    audio: np.ndarray, svd_energy_ratio=0.98, window_size=1024, hop_length=512
) -> np.ndarray:
    """
    Apply SVD smoothing to audio signal using windowed approach
    Args:
        audio: Input audio signal of shape (samples,)
        svd_energy_ratio: Energy ratio for SVD truncation
        window_size: Size of each window for SVD processing
        hop_length: Hop length between windows
    Returns:
        Smoothed audio signal of same shape
    """
    if len(audio.shape) != 1:
        raise ValueError(f"Expected 1D audio signal, got shape {audio.shape}")

    # Pad audio to ensure we can process all samples
    pad_length = window_size - (len(audio) % hop_length)
    if pad_length < window_size:
        audio_padded = np.pad(audio, (0, pad_length), mode="constant")
    else:
        audio_padded = audio

    # Create windowed matrix for SVD
    n_frames = (len(audio_padded) - window_size) // hop_length + 1
    windowed_matrix = np.zeros((window_size, n_frames))

    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + window_size
        windowed_matrix[:, i] = audio_padded[start_idx:end_idx]

    # Apply SVD
    U, s, Vt = svd(windowed_matrix, full_matrices=False)

    # Determine number of components to keep
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    n_components = np.argmax(cumulative_energy >= svd_energy_ratio) + 1
    n_components = max(1, min(n_components, len(s)))

    # Reconstruct with limited components
    s_truncated = s[:n_components]
    U_truncated = U[:, :n_components]
    Vt_truncated = Vt[:n_components, :]

    # Reconstruct the smoothed windowed matrix
    smoothed_windowed = U_truncated @ np.diag(s_truncated) @ Vt_truncated

    # Reconstruct audio signal using overlap-add
    smoothed_audio = np.zeros(len(audio_padded))
    window_count = np.zeros(len(audio_padded))

    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + window_size
        smoothed_audio[start_idx:end_idx] += smoothed_windowed[:, i]
        window_count[start_idx:end_idx] += 1

    # Normalize by window count to handle overlaps
    smoothed_audio = smoothed_audio / np.maximum(window_count, 1)

    # Return original length
    return smoothed_audio[: len(audio)]


# def apply_svd_smoothing_audio_simple(
#      audio: np.ndarray, svd_energy_ratio=0.95
# ) -> np.ndarray:
#      """
#      Apply SVD smoothing to audio signal using simple matrix reshaping
#      # Args:
#          audio: Input audio signal of shape (samples,)
#          svd_energy_ratio: Energy ratio for SVD truncation
#      Returns:
#          Smoothed audio signal of same shape
#      """
#      if len(audio.shape) != 1:
#          raise ValueError(f"Expected 1D audio signal, got shape {audio.shape}")
#
#      # Reshape audio into a matrix for SVD
#      # We'll create a matrix where each row is a shifted version of the signal
#      # This creates a Hankel-like matrix structure
#      window_size = min(1024, len(audio) // 4)  # Adaptive window size
#      if window_size < 2:
#          return audio  # Signal too short for meaningful SVD
#
#      n_rows = len(audio) - window_size + 1
#      if n_rows <= 0:
#          return audio
#
#      # Create the matrix
#      matrix = np.zeros((n_rows, window_size))
#      for i in range(n_rows):
#          matrix[i, :] = audio[i : i + window_size]
#
#      # Apply SVD
#      U, s, Vt = svd(matrix, full_matrices=False)
#
#      # Determine number of components to keep
#      cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
#      n_components = np.argmax(cumulative_energy >= svd_energy_ratio) + 1
#      n_components = max(1, min(n_components, len(s)))
#
#      # Reconstruct with limited components
#      s_truncated = s[:n_components]
#      U_truncated = U[:, :n_components]
#      Vt_truncated = Vt[:n_components, :]
#
#      # Reconstruct the smoothed matrix
#      smoothed_matrix = U_truncated @ np.diag(s_truncated) @ Vt_truncated
#
#      # Reconstruct audio by averaging overlapping elements
#      smoothed_audio = np.zeros(len(audio))
#      count = np.zeros(len(audio))
#
#      for i in range(n_rows):
#          smoothed_audio[i : i + window_size] += smoothed_matrix[i, :]
#          count[i : i + window_size] += 1
#
#      # Normalize by count
#      smoothed_audio = smoothed_audio / np.maximum(count, 1)
#
#      return smoothed_audio


def get_svd_info_audio(
    audio: np.ndarray,
    svd_energy_ratio=0.98,
) -> Dict[str, Any]:
    """
    Get information about SVD decomposition for audio analysis
    Args:
        audio: Input audio signal of shape (samples,)
        svd_energy_ratio: Energy ratio for SVD truncation
        method: 'simple' or 'windowed' approach
    Returns:
        Dictionary with SVD analysis information
    """
    window_size = 1024
    hop_length = 512
    n_frames = (len(audio) - window_size) // hop_length + 1
    if n_frames <= 0:
        return {"error": "Signal too short for SVD analysis"}

    matrix = np.zeros((window_size, n_frames))
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + window_size
        if end_idx <= len(audio):
            matrix[:, i] = audio[start_idx:end_idx]

    # Perform SVD
    U, s, Vt = svd(matrix, full_matrices=False)

    # Calculate energy ratios
    energy_ratios = np.cumsum(s**2) / np.sum(s**2)

    # Determine actual components used
    n_components = np.argmax(energy_ratios >= svd_energy_ratio) + 1
    n_components = max(1, min(n_components, len(s)))

    return {
        "total_components": len(s),
        "used_components": n_components,
        "energy_preserved": energy_ratios[n_components - 1],
        "singular_values": s,
        "compression_ratio": n_components / len(s),
        "matrix_shape": matrix.shape,
        "window_size": window_size,
    }


if __name__ == "__main__":
    BASE_FOLDER = os.path.join(
        "datasets", "coughvid"  # Update this path to your audio files
    )
    AUDIO_FOLDER = os.path.join(
        BASE_FOLDER, "wav"  # Update this path to your audio files
    )

    # Look for common audio file extensions
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
    all_file_paths = []
    for ext in audio_extensions:
        all_file_paths.extend(glob.glob(os.path.join(AUDIO_FOLDER, ext)))

    if not all_file_paths:
        print(f"No audio files found in {AUDIO_FOLDER}")
        print("Please update the AUDIO_FOLDER path to point to your audio files.")
        exit()

    # --- NEW: Define the list of specific filenames to process ---
    # This list should be populated with the filenames (e.g., 'audio1.wav') to be processed.
    # You can load this from a text file, a CSV column, or define it manually.
    entire_wav_cough_filenames = np.load(os.path.join("datasets", "covidUK", "entire_wav_cough_filenames.npy"))


    if len(entire_wav_cough_filenames) == 0:
        print("Warning: The 'entire_wav_cough_filenames' list is empty. Processing all found audio files.")
        file_paths = all_file_paths
    else:
        file_paths = entire_wav_cough_filenames

    # Create output folder for processed audio
    output_folder = os.path.join(BASE_FOLDER, "wav_smooth")
    os.makedirs(output_folder, exist_ok=True)

    svd_energy_ratio = 0.98

    print(f"Found {len(file_paths)} audio files to process.")
    print(f"Processing with SVD energy ratio: {svd_energy_ratio}")
    print(f"Output folder: {output_folder}")
    print("=" * 50)

    # Initialize counters for summary
    successful_files = 0
    failed_files = 0
    skipped_files = 0
    failed_file_names = []

    # Process files with progress bar
    for f in tqdm(file_paths, desc="Processing audio files", unit="file"):
        try:
            # Define output path for the processed file
            output_path = os.path.join(output_folder, os.path.basename(f))

            # --- ADDED CHECK: Skip if the file already exists ---
            if os.path.exists(output_path):
                skipped_files += 1
                continue  # Move to the next file

            # Load audio file
            audio, sr = librosa.load(f, sr=None)

            # Get SVD analysis info (uncomment if needed)
            # audio_info = get_svd_info_audio(audio, svd_energy_ratio=svd_energy_ratio)
            # tqdm.write(f"SVD Info for {os.path.basename(f)}: {audio_info}")

            # Apply SVD smoothing
            smoothed_audio = apply_svd_smoothing_audio(
                audio, svd_energy_ratio=svd_energy_ratio
            )

            # Save processed audio
            sf.write(output_path, smoothed_audio, sr)

            successful_files += 1

            # Optional: Update progress bar description with current file
            # tqdm.write(f"✓ Processed: {os.path.basename(f)}")

        except Exception as e:
            failed_files += 1
            failed_file_names.append(os.path.basename(f))
            tqdm.write(f"✗ Error processing {os.path.basename(f)}: {e}")
            sf.write(output_path, audio, sr)


    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files considered for processing: {len(file_paths)}")
    print(f"Successfully processed: {successful_files}")
    print(f"Skipped (already exist): {skipped_files}")
    print(f"Failed to process: {failed_files}")

    if failed_files > 0:
        print(f"\nFailed files:")
        for failed_file in failed_file_names:
            print(f"  - {failed_file}")

    print(f"\nProcessed files saved to: {output_folder}")
