import argparse
import numpy as np
from tqdm import tqdm
from src.util import get_entire_signal_librosa
import os
from glob import glob

# for pretraining


def preprocess_spectrogram_SSL(
    modality="modality", input_sec=2, form="", nfft=1024, hop=512, TRIM=True
):

    path = "datasets/coughvid/"

    output_dir = f"datasets/coughvid/entire_spec_npy{form}/"
    os.makedirs(output_dir, exist_ok=True)

    filenames = glob(path + f"wav{form}/*.wav")

    filenames = [f"{os.path.basename(f)}" for f in filenames]
    invalid_data = 0

    filename_list = []

    # use metadata as outer loop to enable quality check
    for file in tqdm(filenames):

        userID = file.split(".")[0]

        if os.path.exists(path + f"wav{form}/" + file):
            data = get_entire_signal_librosa(
                f"datasets/coughvid/wav{form}",
                userID,
                spectrogram=True,
                input_sec=input_sec,
                nfft=nfft,
                hop=hop,
                TRIM=TRIM,
            )

            if data is None:
                invalid_data += 1
                continue

            # saving to individual npy files
            np.save(output_dir + userID + ".npy", data)
            filename_list.append(f"datasets/coughvid/entire_spec_npy{form}/" + userID)

    np.save(f"datasets/coughvid/entire_spec_filenames{form}.npy", filename_list)
    print(
        "finished preprocessing cough: valid data",
        len(filename_list),
        "; invalid data",
        invalid_data,
    )


# finished preprocessing cough: valid data 7179 ; invalid data 327


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="cough")
    parser.add_argument("--input_sec", type=int, default=0)
    parser.add_argument("--form", type=str, default="_segmented_phase_2")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--TRIM", type=bool, default=True)

    args = parser.parse_args()

    preprocess_spectrogram_SSL(
        modality=args.modality,
        input_sec=args.input_sec,
        form=args.form,
        nfft=args.nfft,
        hop=args.hop,
        TRIM=args.TRIM,
    )
