import os
import wave
import contextlib
import numpy as np

folder_path = os.path.join("datasets", "coughvid", "wav_segmented")

durations = []

if not os.path.isdir(folder_path):
    print("Error: The specified path is not a valid directory.")
else:
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            try:
                with contextlib.closing(wave.open(file_path, "r")) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    if rate > 0:
                        duration = frames / float(rate)
                        durations.append(duration)
                    else:
                        print(
                            f"Warning: Could not get frame rate for {filename}. Skipping."
                        )
            except wave.Error as e:
                print(f"Warning: Could not process {filename}. Reason: {e}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred with {filename}: {e}. Skipping.")

    if durations:
        file_count = len(durations)
        mean_duration = np.mean(durations)
        lower_quantile = np.quantile(durations, 0.025)
        upper_quantile = np.quantile(durations, 0.975)

        print(f"\nNumber of WAV files found: {file_count}")
        print(
            f"Mean duration [95% quantile range]: {mean_duration:.2f} [{lower_quantile:.2f} - {upper_quantile:.2f}] seconds"
        )
    else:
        print("No .wav files were found in the specified directory.")
