import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
import torchaudio
import librosa
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement SpecAugment for wav2vec2 pretraining.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = np.zeros((batch_size, sequence_length), dtype=bool)

    for i in range(batch_size):
        # randomly choose indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(sequence_length - (mask_length - 1)),
            num_masked_spans,
            replace=False,
        )

        # fill mask
        for j in spec_aug_mask_idx:
            mask[i, j : j + mask_length] = True

    return mask


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Ensure we have attention_mask
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_values"])

        return batch


class Wav2VecDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, target_sr=16000, max_duration_s=10):
        self.audio_paths = audio_paths
        self.target_sr = target_sr
        self.max_len = target_sr * max_duration_s
        # Filter out non-existent files
        self.valid_paths = []
        for path in audio_paths:
            if os.path.exists(path) or os.path.exists(path.replace(".npy", ".wav")):
                self.valid_paths.append(path)
        if len(self.valid_paths) < len(audio_paths):
            print(
                f"Warning: {len(audio_paths) - len(self.valid_paths)} files not found, using {len(self.valid_paths)} valid files"
            )
        self.audio_paths = self.valid_paths if self.valid_paths else audio_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]

        # Try to load actual audio file
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # Trim or pad to max length
        if len(waveform) > self.max_len:
            # Random crop for augmentation
            start_idx = np.random.randint(0, len(waveform) - self.max_len + 1)
            waveform = waveform[start_idx : start_idx + self.max_len]
        else:
            padding_needed = self.max_len - len(waveform)
            waveform = np.pad(waveform, (0, padding_needed), mode="constant")

        return {"input_values": waveform.astype(np.float32)}


class Wav2Vec2PretrainingModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "facebook/wav2vec2-base",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        num_negatives: int = 100,
        codevector_dim: int = 256,
        num_codevector_groups: int = 2,
        num_codevectors_per_group: int = 320,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize config
        config = Wav2Vec2Config(
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            num_negatives=num_negatives,
            codevector_dim=codevector_dim,
            num_codevector_groups=num_codevector_groups,
            num_codevectors_per_group=num_codevectors_per_group,
            do_stable_layer_norm=True,
            feat_extract_norm="layer",
            # Add these to ensure contrastive learning works
            contrastive_logits_temperature=0.1,
            diversity_loss_weight=0.1,
            proj_codevector_dim=codevector_dim,
            tdnn_dim=[512, 512, 512, 512, 1500],
            tdnn_kernel=[5, 3, 3, 1, 1],
            tdnn_dilation=[1, 2, 3, 1, 1],
            # Ensure feature extractor config is proper
            conv_dim=[512, 512, 512, 512, 512, 512, 512],
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
        )

        # Initialize model for pretraining
        self.model = Wav2Vec2ForPreTraining(config)

        # Feature extractor for preprocessing
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

        # Ensure the model is in training mode for pretraining
        self.model.train()

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional feature extractor
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(
            self.model.config.conv_kernel, self.model.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def forward(self, input_values, attention_mask=None):
        # Ensure input has batch dimension
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        # The model needs proper masking for pretraining
        batch_size, raw_sequence_length = input_values.size()

        # Compute the sequence length after convolutional layers
        sequence_length = self._get_feat_extract_output_lengths(raw_sequence_length)

        # Debug print
        if hasattr(self, "_debug_printed") and not self._debug_printed:
            print(
                f"Debug - Raw sequence length: {raw_sequence_length}, Feature sequence length: {sequence_length}"
            )
            self._debug_printed = True

        # Create mask_time_indices for the model with the correct sequence length
        # Make sure sequence_length is an integer
        if isinstance(sequence_length, torch.Tensor):
            sequence_length = int(sequence_length.item())

        # Only create mask if sequence length is valid
        if sequence_length > self.hparams.mask_time_length:
            mask_time_indices = _compute_mask_indices(
                (
                    batch_size,
                    sequence_length,
                ),  # Use feature sequence length, not raw sequence length
                mask_prob=self.hparams.mask_time_prob,
                mask_length=self.hparams.mask_time_length,
            )
            mask_time_indices = torch.tensor(
                mask_time_indices, dtype=torch.bool, device=input_values.device
            )
        else:
            # If sequence is too short, don't mask
            mask_time_indices = None
            print(
                f"Warning: Sequence length {sequence_length} is too short for masking with length {self.hparams.mask_time_length}"
            )

        # Also update attention mask if provided to match feature sequence length
        if attention_mask is not None:
            # Compute which positions in the feature sequence are padded
            attention_mask = self._get_feature_vector_attention_mask(
                sequence_length, attention_mask
            )

        return self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        """
        Computes the attention mask for the feature vectors
        """
        # Basically, the convolutional layers downsample the input by a factor of 160
        # So we need to take every 160th position from the attention mask
        # This is a simplified version - you might need to adjust based on your specific model config

        output_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        print(outputs)
        # For wav2vec2 pretraining, we need to handle the loss components
        loss = outputs.loss

        # If loss is None, there might be an issue with the batch
        if loss is None:
            # Check if we have the loss components
            if (
                hasattr(outputs, "contrastive_loss")
                and outputs.contrastive_loss is not None
            ):
                loss = outputs.contrastive_loss
                if (
                    hasattr(outputs, "diversity_loss")
                    and outputs.diversity_loss is not None
                ):
                    loss = loss + outputs.diversity_loss
            else:
                print(f"Warning: No loss computed at batch {batch_idx}")
                # Skip this batch by returning None
                return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log additional metrics if available
        if (
            hasattr(outputs, "contrastive_loss")
            and outputs.contrastive_loss is not None
        ):
            self.log("train_contrastive_loss", outputs.contrastive_loss, on_step=True)
        if hasattr(outputs, "diversity_loss") and outputs.diversity_loss is not None:
            self.log("train_diversity_loss", outputs.diversity_loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss

        # Check if loss is None
        if loss is None:
            print(f"Warning: Validation loss is None at batch {batch_idx}")
            return

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warmup scheduler
        def lr_lambda(current_step: int):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def extract_features(self, input_values):
        """Extract features from audio for downstream tasks"""
        with torch.no_grad():
            outputs = self.model.wav2vec2(
                input_values=input_values,
            )
            features = outputs.last_hidden_state
        return features


class CombinedWav2VecDataset(torch.utils.data.Dataset):
    """Dataset that combines multiple datasets with proper indexing"""

    def __init__(self, datasets, dataset_names):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side="right")
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        item = self.datasets[dataset_idx][local_idx]
        return item


def train_wav2vec2(
    title: str,
    data_source: Dict[str, int],
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
):
    """Main training function for Wav2Vec2"""

    print(f"Training Wav2Vec2 with data sources: {data_source}")

    # Disable CUDNN for conv1d to avoid the warning
    torch.backends.cudnn.enabled = False

    # Create datasets
    train_datasets = []
    val_datasets = []
    dataset_names = []

    for dt, max_duration in data_source.items():
        print(f"Loading {dt} dataset...")

        if dt in ["covidbreath", "covidcough"]:
            modality = dt[5:]
            filenames = list(
                np.load(
                    f"datasets/covid19-sounds/SSL_entireaudio_filenames_{modality}.npy"
                )
            )
        elif dt == "icbhi":
            icbhi_filenames = np.load("datasets/icbhi/entire_spec_filenames.npy")
            train_test = np.load("datasets/icbhi/entire_spec_split.npy")
            filenames = list(icbhi_filenames[train_test == "train"])
        elif dt == "coughvid":
            filenames = list(
                np.load("datasets/coughvid/entire_wav_cough_filenames.npy")
            )
        elif dt == "hf_lung":
            filenames = list(np.load("datasets/hf_lung/entire_spec_filenames.npy"))
        elif dt == "covidUKexhalation":
            filenames = list(
                np.load("datasets/covidUK/entire_exhalation_filenames.npy")
            )
        elif dt == "covidUKcough":
            filenames = list(np.load("datasets/covidUK/entire_wav_cough_filenames.npy"))
        else:
            continue

        # Split data
        train_files, val_files = train_test_split(
            filenames, test_size=0.1, random_state=1337
        )

        # Create datasets
        train_dataset = Wav2VecDataset(
            train_files, target_sr=16000, max_duration_s=max_duration
        )
        val_dataset = Wav2VecDataset(
            val_files, target_sr=16000, max_duration_s=max_duration
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        dataset_names.append(dt)

        print(f"{dt}: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Combine datasets
    combined_train = CombinedWav2VecDataset(train_datasets, dataset_names)
    combined_val = CombinedWav2VecDataset(val_datasets, dataset_names)

    # Create model
    model = Wav2Vec2PretrainingModel(learning_rate=learning_rate)

    # Create data collator
    data_collator = DataCollatorForWav2Vec2Pretraining(
        feature_extractor=model.feature_extractor,
        padding=True,
    )

    # Create dataloaders with fewer workers to avoid issues
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,  # Reduced from 4
        pin_memory=True,
    )

    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,  # Reduced from 4
        pin_memory=True,
    )

    # Setup logging and checkpoints
    logger = CSVLogger(
        save_dir="cks/logs",
        name="wav2vec2",
        version=title,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=f"cks/model/wav2vec2/{title}",
        filename="wav2vec2-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        every_n_epochs=10,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Effective batch size = 32 * 4 = 128
        precision=16,  # Use mixed precision for memory efficiency
    )

    # Train model
    print("Starting Wav2Vec2 pretraining...")
    trainer.fit(model, train_loader, val_loader)

    # Test model
    print("Testing model...")
    trainer.test(model, val_loader)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, required=True)

    # Data sources
    parser.add_argument("--covidbreath", action="store_true")
    parser.add_argument("--covidcough", action="store_true")
    parser.add_argument("--icbhi", action="store_true")
    parser.add_argument("--coughvid", action="store_true")
    parser.add_argument("--hf_lung", action="store_true")
    parser.add_argument("--covidUKexhalation", action="store_true")
    parser.add_argument("--covidUKcough", action="store_true")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define max durations (in seconds) for each dataset
    optimal_max_duration = {
        "covidbreath": 8,
        "covidcough": 2,
        "icbhi": 2,
        "coughvid": 2,
        "hf_lung": 8,
        "covidUKexhalation": 4,
        "covidUKcough": 2,
    }

    # Collect selected data sources
    data_source = {}
    for dt, max_duration in optimal_max_duration.items():
        if getattr(args, dt):
            data_source[dt] = max_duration

    # Train model
    train_wav2vec2(
        title=args.title,
        data_source=data_source,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
