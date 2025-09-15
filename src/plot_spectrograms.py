import os
import numpy as np
import matplotlib.pyplot as plt


def convert_npy_to_png(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith(".npy"):  # Process only .npy files
            npy_path = os.path.join(source_folder, file_name)

            spectrogram = np.load(npy_path)

            spectrogram = spectrogram.T

            png_file_name = os.path.splitext(file_name)[0] + ".png"
            png_path = os.path.join(output_folder, png_file_name)

            plt.figure()
            plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(label="Intensity")
            plt.title(f"Spectrogram: {file_name}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()

            print(f"Created {png_path}")


def plot_single_spectrogram(file_name, output_folder):
    if file_name.endswith(".npy"):  # Process only .npy files

        spectrogram = np.load(file_name)

        spectrogram = spectrogram.T

        png_file_name = os.path.basename(file_name) + ".png"
        png_path = os.path.join(output_folder, png_file_name)

        plt.figure()
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Intensity")
        plt.title(f"Spectrogram: {file_name}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        print(f"Created {png_path}")


if __name__ == "__main__":
    source_folder = os.path.join("datasets", "coughvid", "entire_spec_npy_segmented")
    output_folder = os.path.join("generated", "spectrograms", "phases", "phases1")
    # convert_npy_to_png(source_folder, output_folder)

    os.makedirs(output_folder, exist_ok=True)

    file = "datasets/coughvid/entire_spec_npy_segmented_phase_1/3c2443c6-efce-48dd-906d-21a2e8acd29b_cough_6.npy"
    plot_single_spectrogram(file, output_folder)
