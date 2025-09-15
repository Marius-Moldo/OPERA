import os
import matplotlib.pyplot as plt
import numpy as np
import wave

# Path to your folder with wav files
input_folder = os.path.join("datasets", "coughvid", "wav_segmented")
output_folder = os.path.join("generated", "wav_plots")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through files in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(input_folder, filename)

        # Read wav file
        with wave.open(file_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:  # 32-bit audio
            audio_np = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo (plot only one channel)
        if n_channels > 1:
            audio_np = audio_np[::n_channels]

        # Time axis
        time_axis = np.linspace(0, len(audio_np) / framerate, num=len(audio_np))

        # Plot waveform
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio_np, color="blue")
        plt.title(f"Waveform of {filename}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            output_folder, f"{os.path.splitext(filename)[0]}.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"Waveform image saved to: {output_path}")

print("Waveform images saved successfully.")
import os
import matplotlib.pyplot as plt
import numpy as np
import wave

# Path to your folder with wav files
input_folder = os.path.join("data", "CoughDataset", "default")
output_folder = os.path.join("generated", "wav_plots")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through files in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(input_folder, filename)

        # Read wav file
        with wave.open(file_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:  # 32-bit audio
            audio_np = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo (plot only one channel)
        if n_channels > 1:
            audio_np = audio_np[::n_channels]

        # Time axis
        time_axis = np.linspace(0, len(audio_np) / framerate, num=len(audio_np))

        # Plot waveform
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio_np, color="blue")
        plt.title(f"Waveform of {filename}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            output_folder, f"{os.path.splitext(filename)[0]}.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"Waveform image saved to: {output_path}")

print("Waveform images saved successfully.")
