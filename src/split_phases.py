import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob


def compute_spectral_slope(y, sr, n_fft=2048, hop_length=512):
    """
    Compute spectral slope for each frame of the audio.

    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames

    Returns:
        spectral_slopes: Array of spectral slopes for each frame
        times: Time values for each frame
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute spectral slope for each frame
    spectral_slopes = []

    for frame_idx in range(magnitude.shape[1]):
        frame_mag = magnitude[:, frame_idx]

        # Avoid log of zero
        frame_mag = np.maximum(frame_mag, 1e-10)

        # Compute weighted linear regression for spectral slope
        # Using frequency as x and log magnitude as y
        log_mag = np.log(frame_mag)

        # Simple linear regression
        x = freqs
        y = log_mag

        # Remove DC component
        x = x[1:]
        y = y[1:]

        # Compute slope using least squares
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        spectral_slopes.append(slope)

    spectral_slopes = np.array(spectral_slopes)
    times = librosa.frames_to_time(
        np.arange(len(spectral_slopes)), sr=sr, hop_length=hop_length
    )

    return spectral_slopes, times


def find_phase_boundary(spectral_slopes, smooth_window=5):
    """
    Find the phase boundary by locating the most negative spectral slope.

    Args:
        spectral_slopes: Array of spectral slopes
        smooth_window: Window size for smoothing

    Returns:
        boundary_idx: Index of the phase boundary (most negative slope)
        spectral_slopes_smooth: Smoothed spectral slopes
    """
    # Smooth the spectral slopes
    if smooth_window > 1:
        spectral_slopes_smooth = savgol_filter(
            spectral_slopes, window_length=smooth_window, polyorder=2
        )
    else:
        spectral_slopes_smooth = spectral_slopes

    # Find the point with the most negative (minimum) spectral slope
    boundary_idx = np.argmin(spectral_slopes_smooth)

    return boundary_idx, spectral_slopes_smooth


def find_highest_peak(y, sr):
    """
    Find the highest peak in the audio signal.

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        peak_time: Time of the highest peak
        peak_amplitude: Amplitude of the highest peak
        peak_idx: Sample index of the highest peak
    """
    # Find the absolute maximum (considering both positive and negative peaks)
    abs_y = np.abs(y)
    peak_idx = np.argmax(abs_y)
    peak_amplitude = y[peak_idx]
    peak_time = peak_idx / sr

    return peak_time, peak_amplitude, peak_idx


def find_phase_boundary_time(audio_path, smooth_window=5):
    """
    Find the phase boundary time and highest peak for a given audio file.

    Args:
        audio_path: Path to the .wav file
        smooth_window: Window size for smoothing spectral slopes

    Returns:
        boundary_time: Time point of the phase boundary
        peak_time: Time of the highest peak
        peak_amplitude: Amplitude of the highest peak
        y: Audio samples
        sr: Sample rate
        spectral_slopes: Array of spectral slopes
        times: Time values for each frame
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Compute spectral slopes
    spectral_slopes, times = compute_spectral_slope(y, sr)

    # Find phase boundary
    boundary_idx, spectral_slopes_smooth = find_phase_boundary(
        spectral_slopes, smooth_window
    )

    # Convert boundary index to time
    boundary_time = times[boundary_idx]

    # Find highest peak
    peak_time, peak_amplitude, peak_idx = find_highest_peak(y, sr)

    return (
        boundary_time,
        peak_time,
        peak_amplitude,
        y,
        sr,
        spectral_slopes_smooth,
        times,
    )


def create_phase_visualization(
    audio_folder, output_dir="./phase_visualizations", max_files=10
):
    """
    Create phase boundary and peak visualizations for the first N wav files in a folder.

    Args:
        audio_folder: Path to folder containing .wav files
        output_dir: Directory to save the visualization plots
        max_files: Maximum number of files to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all wav files in the folder
    wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))

    if not wav_files:
        print(f"No .wav files found in {audio_folder}")
        return

    # Limit to first max_files
    wav_files = wav_files[:max_files]

    print(f"Processing {len(wav_files)} wav files...")

    for i, audio_path in enumerate(wav_files):
        try:
            print(
                f"Processing file {i + 1}/{len(wav_files)}: {os.path.basename(audio_path)}"
            )

            # Find phase boundary and peak
            boundary_time, peak_time, peak_amplitude, y, sr, spectral_slopes, times = (
                find_phase_boundary_time(audio_path)
            )

            # Create visualization with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot waveform (top subplot)
            time_audio = np.arange(len(y)) / sr
            ax1.plot(time_audio, y, "b-", alpha=0.7, linewidth=1)

            # Add phase boundary line to waveform
            ax1.axvline(
                x=boundary_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Phase boundary at {boundary_time:.3f}s",
            )

            # Add highest peak marker
            ax1.plot(
                peak_time,
                peak_amplitude,
                "o",
                color="orange",
                markersize=10,
                markeredgecolor="darkred",
                markeredgewidth=2,
                label=f"Highest peak at {peak_time:.3f}s (amp: {peak_amplitude:.3f})",
                zorder=5,
            )

            # Add vertical line at peak location
            ax1.axvline(
                x=peak_time,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
            )

            # Styling for waveform
            ax1.set_xlabel("Time (s)", fontsize=12)
            ax1.set_ylabel("Amplitude", fontsize=12)
            ax1.set_title(
                f"Cough Phase Boundary & Peak - {os.path.basename(audio_path)}",
                fontsize=14,
            )
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Add phase labels to waveform
            mid_phase1 = boundary_time / 2
            mid_phase2 = boundary_time + (time_audio[-1] - boundary_time) / 2

            # Determine which phase the peak is in
            peak_in_phase1 = peak_time < boundary_time

            ax1.text(
                mid_phase1,
                ax1.get_ylim()[1] * 0.8,
                f"Phase 1{'*' if peak_in_phase1 else ''}",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightblue" if not peak_in_phase1 else "gold",
                    alpha=0.7,
                ),
            )
            ax1.text(
                mid_phase2,
                ax1.get_ylim()[1] * 0.8,
                f"Phase 2{'*' if not peak_in_phase1 else ''}",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightgreen" if peak_in_phase1 else "gold",
                    alpha=0.7,
                ),
            )

            # Add note about which phase contains the peak
            ax1.text(
                0.02,
                0.95,
                f"* Peak in {'Phase 1' if peak_in_phase1 else 'Phase 2'}",
                transform=ax1.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                verticalalignment="top",
            )

            # Plot spectral slopes (bottom subplot)
            ax2.plot(
                times, spectral_slopes, "g-", linewidth=1.5, label="Spectral Slope"
            )
            ax2.axvline(
                x=boundary_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Most negative slope at {boundary_time:.3f}s",
            )
            ax2.axvline(
                x=peak_time,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=f"Peak location at {peak_time:.3f}s",
            )
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

            # Highlight the most negative point
            boundary_idx = np.argmin(spectral_slopes)
            ax2.plot(
                boundary_time,
                spectral_slopes[boundary_idx],
                "ro",
                markersize=8,
                label=f"Min slope: {spectral_slopes[boundary_idx]:.4f}",
            )

            # Styling for spectral slopes
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_ylabel("Spectral Slope", fontsize=12)
            ax2.set_title("Spectral Slope Over Time", fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(
                output_dir, f"{base_name}_phase_boundary_peak.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Saved visualization: {output_path}")
            print(f"  Phase boundary at: {boundary_time:.3f} seconds")
            print(
                f"  Highest peak at: {peak_time:.3f} seconds (amplitude: {peak_amplitude:.3f})"
            )
            print(
                f"  Peak is in: {'Phase 1' if peak_time < boundary_time else 'Phase 2'}"
            )
            print(
                f"  Most negative spectral slope: {spectral_slopes[boundary_idx]:.4f}"
            )
            print(f"  Total duration: {len(y) / sr:.3f} seconds")
            print()

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

    print(f"Visualization complete! Files saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Process first 10 wav files in the audio folder
    audio_folder = os.path.join("datasets", "coughvid", "wav")
    create_phase_visualization(
        audio_folder, output_dir=os.path.join("generated", "phases"), max_files=20
    )
