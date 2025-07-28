import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import librosa
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.benchmark.baseline.vggish.mel_features import log_mel_spectrogram


class Wav2VecDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, target_sr=16000, max_duration_s=10):
        self.audio_paths = audio_paths
        self.target_sr = target_sr
        self.max_len = target_sr * max_duration_s

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # Trim or pad to max length
        if len(waveform) > self.max_len:
            start_idx = np.random.randint(0, len(waveform) - self.max_len + 1)
            waveform = waveform[start_idx: start_idx + self.max_len]
        else:
            padding_needed = self.max_len - len(waveform)
            waveform = np.pad(waveform, (0, padding_needed), mode="constant")

        return {"input_values": waveform.astype(np.float32)}


class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.
    """

    def __init__(self, model, feature_extractor, padding="longest", pad_to_multiple_of=None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # Make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # Make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # Compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # Sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask")
        )

        # Sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


class Wav2Vec2PretrainingModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 5e-5):
        super().__init__()
        self.save_hyperparameters()

        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")

        # Important: Configure masking for contrastive loss
        config.mask_time_prob = 0.05  # Probability of masking a time step
        config.mask_time_length = 10  # Length of each mask
        config.mask_feature_prob = 0.0  # Start with feature masking disabled
        config.contrastive_logits_temperature = 0.1
        config.diversity_loss_weight = 0.1
        config.num_negatives = 100  # Number of negative samples for contrastive loss
        config.do_stable_layer_norm = True
        config.feat_extract_norm = "group"

        self.model = Wav2Vec2ForPreTraining(config)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        # Create data collator
        self.data_collator = DataCollatorForWav2Vec2Pretraining(
            model=self.model,
            feature_extractor=self.feature_extractor,
            pad_to_multiple_of=None
        )

        # For Gumbel temperature scheduling
        self.max_gumbel_temperature = 2.0
        self.min_gumbel_temperature = 0.5
        self.gumbel_temperature_decay = 0.999995

        # Store config for mask computation
        self.config = config

    def forward(self, input_values, mask_time_indices=None, sampled_negative_indices=None, attention_mask=None):
        return self.model(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        # Forward pass with mask
        outputs = self.forward(**batch)
        loss = outputs.loss

        print(loss)

        # Update Gumbel temperature
        gumbel_temperature = max(
            self.max_gumbel_temperature * self.gumbel_temperature_decay ** self.global_step,
            self.min_gumbel_temperature,
        )
        self.model.set_gumbel_temperature(gumbel_temperature)

        # Calculate additional metrics
        with torch.no_grad():
            # Compute cosine similarity for masked positions
            cosine_sim = torch.cosine_similarity(
                outputs.projected_states,
                outputs.projected_quantized_states,
                dim=-1
            )
            cosine_sim = cosine_sim[batch["mask_time_indices"].bool()].mean()

            # Calculate percent masked
            percent_masked = batch["mask_time_indices"].sum() / batch["mask_time_indices"].numel()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_contrastive_loss", outputs.contrastive_loss, on_step=True)
        self.log("train_diversity_loss", outputs.diversity_loss, on_step=True)
        self.log("train_cosine_sim", cosine_sim * 100, on_step=True)
        self.log("train_perplexity", outputs.codevector_perplexity, on_step=True)
        self.log("gumbel_temperature", gumbel_temperature, on_step=True)
        self.log("percent_masked", percent_masked, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss

        # Calculate metrics
        with torch.no_grad():
            cosine_sim = torch.cosine_similarity(
                outputs.projected_states,
                outputs.projected_quantized_states,
                dim=-1
            )
            cosine_sim = cosine_sim[batch["mask_time_indices"].bool()].mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_contrastive_loss", outputs.contrastive_loss, on_epoch=True)
        self.log("val_diversity_loss", outputs.diversity_loss, on_epoch=True)
        self.log("val_cosine_sim", cosine_sim * 100, on_epoch=True)
        self.log("val_perplexity", outputs.codevector_perplexity, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Add learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.08,  # 8% warmup
            anneal_strategy='linear'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


def train_wav2vec2(title: str, data_source: dict, n_epochs: int = 100,
                   batch_size: int = 16, learning_rate: float = 5e-5):
    """Main training function for Wav2Vec2"""

    print(f"Training Wav2Vec2 with data sources: {data_source}")

    # Collect all audio files
    all_files = []

    for dt, max_duration in data_source.items():
        print(f"Loading {dt} dataset...")

        if dt in ["covidbreath", "covidcough"]:
            modality = dt[5:]
            filenames = list(np.load(f"datasets/covid19-sounds/SSL_entireaudio_filenames_{modality}.npy"))
        elif dt == "icbhi":
            icbhi_filenames = np.load("datasets/icbhi/entire_spec_filenames.npy")
            train_test = np.load("datasets/icbhi/entire_spec_split.npy")
            filenames = list(icbhi_filenames[train_test == "train"])
        elif dt == "coughvid":
            filenames = list(np.load("datasets/coughvid/entire_wav_cough_filenames.npy"))
        elif dt == "hf_lung":
            filenames = list(np.load("datasets/hf_lung/entire_spec_filenames.npy"))
        elif dt == "covidUKexhalation":
            filenames = list(np.load("datasets/covidUK/entire_exhalation_filenames.npy"))
        elif dt == "covidUKcough":
            filenames = list(np.load("datasets/covidUK/entire_wav_cough_filenames.npy"))
        else:
            continue

        all_files.extend(filenames)
        print(f"{dt}: {len(filenames)} files")

    # Split data
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=1337)

    # Create datasets
    train_dataset = Wav2VecDataset(train_files, target_sr=16000, max_duration_s=10)
    val_dataset = Wav2VecDataset(val_files, target_sr=16000, max_duration_s=10)

    # Create model
    model = Wav2Vec2PretrainingModel(learning_rate=learning_rate)

    # Create dataloaders with the data collator
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=model.data_collator,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=model.data_collator,
        num_workers=4,
        pin_memory=True
    )

    # Setup logging and checkpoints
    logger = CSVLogger(save_dir="cks/logs", name="wav2vec2", version=title)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=f"cks/model/wav2vec2/{title}",
        filename="wav2vec2-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        save_last=True
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        gradient_clip_val=1.0,  # Add gradient clipping
        accumulate_grad_batches=4,  # Effective batch size = 16 * 4 = 64
        precision=16,  # Use mixed precision for faster training
    )

    # Train model
    print("Starting Wav2Vec2 pretraining...")
    trainer.fit(model, train_loader, val_loader)

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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    # Define max durations for each dataset
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

    # Set precision for better performance on RTX 4070
    torch.set_float32_matmul_precision('medium')

    # Train model
    train_wav2vec2(
        title=args.title,
        data_source=data_source,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )