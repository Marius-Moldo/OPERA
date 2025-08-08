# -*- coding: utf-8 -*-
import argparse
import numpy as np
from openpyxl.styles.builtins import output
from tqdm import tqdm
from src.util import get_entire_signal_librosa
import os
from glob import glob


### for pretraining
def preprocess_spectrogram_SSL(
    modality="modality", input_sec=2, form="", nfft=512, hop=256, TRIM=True
):

    path = "datasets/covidUK/"
    # path = ''

    output_dir = f"datasets/covidUK/entire_spec_npy{form}/"
    os.makedirs(output_dir, exist_ok=True)

    # filenames = np.load(path + "entire_" + modality + "_filenames.npy")
    filenames = glob(path + f"audio{form}/*.wav")

    filenames = [f"{os.path.basename(f)}" for f in filenames]

    print("SSL training:", len(filenames))

    invalid_data = 0

    filename_list = []

    # use metadata as outer loop to enable quality check
    for file in tqdm(filenames):
        # print(file)
        userID = file.split(".")[0]
        if os.path.exists(path + f"audio{form}/" + file):
            data = get_entire_signal_librosa(
                path + f"audio{form}/",
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
        filename_list.append(f"datasets/covidUK/entire_spec_npy{form}/" + userID)

    np.save(path + "entire_" + modality + "_filenames_segmented.npy", filename_list)
    print(
        "finished preprocessing breathing: valid data",
        len(filename_list),
        "; invalid data",
        invalid_data,
    )


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
