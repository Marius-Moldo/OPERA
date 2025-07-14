# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/CVxTz/COLA_pytorch
from autrainer.augmentations import SpecAugment
import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
import librosa
from src.model.models_wav2vec2 import Wav2Vec2Pretrainer, Wav2Vec2MD

def combine_dataloaders(dataloaders, train=False):
    if train:
        return CombinedLoader(dataloaders, 'max_size_cycle')
    return CombinedLoader(dataloaders, "sequential")


output_dir = 'spectrogram_images'
os.makedirs(output_dir, exist_ok=True)


# Fixed Wav2Vec2Dataset class
class Wav2Vec2Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, target_sr=16000, max_duration_s=10):
        self.audio_paths = audio_paths
        self.target_sr = target_sr
        self.max_len = target_sr * max_duration_s

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]

        # Load audio file
        waveform, sr = librosa.load(audio_path + '.wav', sr=self.target_sr, mono=True)

        # Ensure waveform is 1D
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        # Truncate or pad to fixed length
        if len(waveform) > self.max_len:
            waveform = waveform[:self.max_len]
        else:
            padding_needed = self.max_len - len(waveform)
            waveform = np.pad(waveform, (0, padding_needed), mode='constant')

        # Convert to tensor and ensure correct shape
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Ensure it's 1D (sequence_length,) - no extra dimensions
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        return waveform


# Fixed _calculate_loss method in Wav2Vec2MD class
def _calculate_loss(self, batch, batch_idx, mode, dataset_idx=None):
    """
    Calculate loss for a given batch (similar to ColaMD's _calculate_loss).
    """
    # Handle different batch formats
    if isinstance(batch, dict):
        input_values = batch['input_values']
    else:
        input_values = batch

    # Ensure input_values has the correct shape
    # Expected: (batch_size, sequence_length)
    if input_values.ndim == 3 and input_values.shape[-1] == 1:
        # Remove the last dimension if it's 1: (batch_size, sequence_length, 1) -> (batch_size, sequence_length)
        input_values = input_values.squeeze(-1)
    elif input_values.ndim == 1:
        # Add batch dimension if missing: (sequence_length,) -> (1, sequence_length)
        input_values = input_values.unsqueeze(0)

    # Ensure the tensor is float32
    input_values = input_values.float()

    # Debug print to check shapes (remove after fixing)
    print(f"Input shape: {input_values.shape}")

    # Forward pass through wav2vec 2.0
    outputs = self.model(input_values)

    # Get the main loss (contrastive + diversity)
    loss = outputs.loss

    # Create log suffix for dataset-specific logging
    log_suffix = f"_{dataset_idx}" if dataset_idx is not None else ""

    # Log main loss
    self.log(f"{mode}_loss{log_suffix}", loss, batch_size=self.batch_size)

    # Log component losses if available
    if hasattr(outputs, 'contrastive_loss'):
        self.log(f"{mode}_contrastive_loss{log_suffix}", outputs.contrastive_loss, batch_size=self.batch_size)
    if hasattr(outputs, 'diversity_loss'):
        self.log(f"{mode}_diversity_loss{log_suffix}", outputs.diversity_loss, batch_size=self.batch_size)

    # Log perplexity if available (indicates codebook usage)
    if hasattr(outputs, 'codevector_perplexity'):
        self.log(f"{mode}_perplexity{log_suffix}", outputs.codevector_perplexity, batch_size=self.batch_size)

    return loss


# Alternative: Custom collate function to ensure proper batching
def wav2vec2_collate_fn(batch):
    """
    Custom collate function to ensure proper tensor shapes for wav2vec2.
    """
    # Stack all waveforms in the batch
    waveforms = torch.stack(batch)

    # Ensure 2D shape: (batch_size, sequence_length)
    if waveforms.ndim == 3 and waveforms.shape[-1] == 1:
        waveforms = waveforms.squeeze(-1)

    return waveforms.float()


# Updated DataLoader creation in train_multiple_data function
def create_dataloaders_fixed(filenames, batch_size):
    """
    Create DataLoaders with proper collate function.
    """
    train, test = train_test_split(filenames, test_size=0.1, random_state=1337)

    train_data = Wav2Vec2Dataset(train)
    val_data = Wav2Vec2Dataset(test)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        collate_fn=wav2vec2_collate_fn  # Add custom collate function
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        collate_fn=wav2vec2_collate_fn  # Add custom collate function
    )

    return train_loader, val_loader



class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def train_multiple_data(title, data_source={"covidbreath": 251},  strategy="crop", n_epoches=512, training_method="wav2vec2"):
    print(data_source)

    method = training_method

    batch_size = 32
    epochs = n_epoches

    print(f"contrastive strategy: {strategy}")

    num_batch = []

    #  constructing dataloaders
    train_loaders, val_loaders = [], []

    print('===============================================================================')
    print('start loading data:')
    for dt, max_len in data_source.items():
        from_npy = True

        if dt == "coughvid":
            filenames = list(
                np.load("datasets/coughvid/entire_wav_cough_filenames_audio.npy"))


        elif dt == "covidUKcough":
            filenames = list(
                np.load("datasets/covidUK/entire_wav_cough_filenames.npy"))

        # # plotting data length distribution
        # data_lengths = []
        # for file_path in filenames:
        #     data = np.load(file_path + ".npy")
        #     data_lengths.append(data.shape[0])

        # import matplotlib.pyplot as plt
        # plt.hist(data_lengths, bins=20, color='skyblue', edgecolor='black')
        # plt.xlabel('Data Length')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Data Length ' + dt)
        # plt.grid(True)
        # plt.savefig("fig/training/{}_length_hist.png".format(dt))
        # plt.clf()

        train, test = train_test_split(
            filenames, test_size=0.1, random_state=1337)

        train_data = Wav2Vec2Dataset(
            train)
        val_data = Wav2Vec2Dataset(
            test)

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=7
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=7
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        print(dt, 'Length of Training, Validation',
              len(train_loader), len(val_loader))
        num_batch.append(len(train_loader))

    print('===============================================================================')
    train_loader = combine_dataloaders(train_loaders, train=True)
    val_loader = combine_dataloaders(val_loaders)

    if training_method == "wav2vec2":
        model = Wav2Vec2MD(
            num_cough_phases=3,  # Learn 3 discrete cough phases
            learning_rate=1e-4,
            num_batch=[258.0, 288, 4, 51, 75, 146, 138],  # Same as ColaMD
            dataset_names=["covidbreath", "covidcough", "icbhi", "coughvid", "hf_lung", "covidUKexhalation",
                           "covidUKcough"]
        )

        logger = CSVLogger(
        save_dir="cks/logs",
        name="combined",
        version=title,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss", mode="min", dirpath="cks/model/combined/" + "_".join(data_source.keys()),
        filename='encoder-' + title +
        '-{epoch:02d}--{valid_acc:.2f}-{valid_loss:.4f}',
        every_n_epochs=50,
        save_top_k=5
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[DecayLearningRate(), checkpoint_callback],
    )

    print('======================SSL Training==============================')
    trainer.fit(model, train_loader, val_loader)
    print('======================SSL Testing==============================')
    trainer.test(dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str)
    parser.add_argument("--data", type=str, default="multiple")

    # for training with multiple data
    parser.add_argument("--covidbreath", type=bool, default=False)
    parser.add_argument("--covidcough", type=bool, default=False)
    parser.add_argument("--icbhi", type=bool, default=False)
    parser.add_argument("--icbhicycle", type=bool, default=False)
    parser.add_argument("--coughvid", type=bool, default=False)
    parser.add_argument("--hf_lung", type=bool, default=False)
    parser.add_argument("--covidUKexhalation", type=bool, default=False)
    parser.add_argument("--covidUKcough", type=bool, default=False)

    parser.add_argument("--strategy", type=str, default="mask")

    # control training
    parser.add_argument("--epoches", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # training goal
    parser.add_argument("--method", type=str, default="wav2vec2")

    args = parser.parse_args()

    optimal_max_len = {"covidbreath": 200, "covidcough": 50, "icbhi": 50, "icbhicycle": 50,
                       "coughvid": 50, "hf_lung": 200, "covidUKexhalation": 100, "covidUKcough": 50}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_source = {}
    for dt, max_len in optimal_max_len.items():
        if getattr(args, dt) is True:
            data_source[dt] = max_len
    train_multiple_data(args.title, data_source=data_source,
                         n_epoches=args.epoches, training_method=args.method, strategy=args.strategy)
