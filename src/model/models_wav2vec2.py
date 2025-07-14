import pytorch_lightning as pl
import torch
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config
import random
import numpy as np


class Wav2Vec2Pretrainer(pl.LightningModule):
    def __init__(
            self,
            num_cough_phases=3,
            learning_rate=1e-4,
            mask_time_prob=0.65,
            mask_time_length=10,
    ):
        """
        A PyTorch Lightning module for wav2vec 2.0 pre-training.
        Args:
            num_cough_phases (int): The number of clusters to learn, corresponding to cough phases.
            learning_rate (float): The learning rate for the optimizer.
            mask_time_prob (float): Probability of masking a given time step.
            mask_time_length (int): Length of each mask span.
        """
        super().__init__()
        # This saves all hyperparameters (num_cough_phases, lr, etc.) to self.hparams
        # and makes them accessible for checkpointing and logging.
        self.save_hyperparameters()

        # 1. Define the wav2vec 2.0 model configuration
        config = Wav2Vec2Config(
            # Standard architecture parameters, can be adjusted
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            # Key parameters for the pre-training task
            mask_time_prob=self.hparams.mask_time_prob,
            mask_time_length=self.hparams.mask_time_length,
            # Set the number of "clusters" to match your desired cough phases
            num_codevector_groups=1,
            num_codevectors_per_group=self.hparams.num_cough_phases,
        )

        # 2. Instantiate the pre-training model from the configuration
        self.model = Wav2Vec2ForPreTraining(config)

    def forward(self, input_values):
        """
        Forward pass through the wav2vec 2.0 model.
        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            Wav2Vec2ForPreTrainingOutput: Contains loss, projected_states, projected_quantized_states, etc.
        """
        return self.model(input_values)

    def training_step(self, batch, batch_idx):
        """
        Training step for wav2vec 2.0 pre-training.
        Args:
            batch: Batch of audio data. Expected to be either:
                   - torch.Tensor of shape (batch_size, sequence_length) for raw audio
                   - dict with 'input_values' key containing the audio tensor
            batch_idx: Index of the current batch
        Returns:
            torch.Tensor: Training loss
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            input_values = batch['input_values']
        else:
            input_values = batch

        # Forward pass - the model automatically handles masking and contrastive learning
        outputs = self.model(input_values)

        # The model returns the contrastive loss for pre-training
        loss = outputs.loss

        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log additional metrics if available
        if hasattr(outputs, 'contrastive_loss'):
            self.log("train_contrastive_loss", outputs.contrastive_loss, on_step=True, on_epoch=True)
        if hasattr(outputs, 'diversity_loss'):
            self.log("train_diversity_loss", outputs.diversity_loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for wav2vec 2.0 pre-training.
        Args:
            batch: Batch of audio data
            batch_idx: Index of the current batch
        Returns:
            torch.Tensor: Validation loss
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            input_values = batch['input_values']
        else:
            input_values = batch

        # Forward pass
        outputs = self.model(input_values)
        loss = outputs.loss

        # Log validation metrics
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log additional metrics if available
        if hasattr(outputs, 'contrastive_loss'):
            self.log("valid_contrastive_loss", outputs.contrastive_loss, on_step=False, on_epoch=True)
        if hasattr(outputs, 'diversity_loss'):
            self.log("valid_diversity_loss", outputs.diversity_loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for wav2vec 2.0 pre-training.
        Args:
            batch: Batch of audio data
            batch_idx: Index of the current batch
        Returns:
            torch.Tensor: Test loss
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            input_values = batch['input_values']
        else:
            input_values = batch

        # Forward pass
        outputs = self.model(input_values)
        loss = outputs.loss

        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Log additional metrics if available
        if hasattr(outputs, 'contrastive_loss'):
            self.log("test_contrastive_loss", outputs.contrastive_loss, on_step=False, on_epoch=True)
        if hasattr(outputs, 'diversity_loss'):
            self.log("test_diversity_loss", outputs.diversity_loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        Returns:
            torch.optim.Optimizer: AdamW optimizer with the specified learning rate
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,  # Adjust based on your training schedule
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def extract_features(self, input_values):
        """
        Extract features from raw audio using the pre-trained wav2vec 2.0 model.
        This method can be used after pre-training to extract representations for downstream tasks.

        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, sequence_length, hidden_size)
        """
        self.model.eval()
        with torch.no_grad():
            # Get the feature extractor and transformer outputs
            outputs = self.model.wav2vec2(input_values)

            # Return the last hidden states (contextualized representations)
            return outputs.last_hidden_state

    def get_quantized_representations(self, input_values):
        """
        Get quantized representations (codebook vectors) from the model.
        Useful for analyzing the learned discrete representations.

        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            tuple: (quantized_features, codevector_perplexity)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_values)
            return outputs.projected_quantized_states, outputs.codevector_perplexity


class Wav2Vec2MD(pl.LightningModule):
    def __init__(
            self,
            num_cough_phases=3,
            learning_rate=1e-4,
            mask_time_prob=0.65,
            mask_time_length=10,
            batch_size=128,
            num_batch=[258.0, 288, 4, 51, 75, 146, 138],  # Dataset sizes similar to ColaMD
            dataset_names=["covidbreath", "covidcough", "icbhi", "coughvid", "hf_lung", "covidUKexhalation",
                           "covidUKcough"],
            diversity_loss_weight=0.1,
            contrastive_loss_weight=1.0,
    ):
        """
        Multi-dataset wav2vec 2.0 pre-training module following ColaMD approach.

        Args:
            num_cough_phases (int): Number of clusters/cough phases to learn
            learning_rate (float): Learning rate for optimizer
            mask_time_prob (float): Probability of masking time steps
            mask_time_length (int): Length of each mask span
            batch_size (int): Batch size for logging
            num_batch (list): Number of samples per dataset for weighted sampling
            dataset_names (list): Names of datasets for logging
            diversity_loss_weight (float): Weight for diversity loss
            contrastive_loss_weight (float): Weight for contrastive loss
        """
        super().__init__()
        self.save_hyperparameters()

        # Store dataset information
        self.batch_size = batch_size
        self.dataset_names = dataset_names
        self.diversity_loss_weight = diversity_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # Calculate normalized weights for dataset sampling (similar to ColaMD)
        print(f"Dataset batch sizes: {num_batch}")
        self.num_batch = [b / np.sum(num_batch) for b in num_batch]
        print(f"Normalized dataset weights: {self.num_batch}")

        # 1. Define the wav2vec 2.0 model configuration
        config = Wav2Vec2Config(
            # Standard architecture parameters
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            # Key parameters for the pre-training task
            mask_time_prob=self.hparams.mask_time_prob,
            mask_time_length=self.hparams.mask_time_length,
            # Set the number of "clusters" to match your desired cough phases
            num_codevector_groups=1,
            num_codevectors_per_group=self.hparams.num_cough_phases,
            # Additional parameters for better multi-dataset training
            diversity_loss_weight=self.diversity_loss_weight,
            contrastive_loss_weight=self.contrastive_loss_weight,
        )

        # 2. Instantiate the pre-training model from the configuration
        self.model = Wav2Vec2ForPreTraining(config)

    def forward(self, input_values):
        """
        Forward pass through the wav2vec 2.0 model.
        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            Wav2Vec2ForPreTrainingOutput: Contains loss, projected_states, projected_quantized_states, etc.
        """
        return self.model(input_values)

    def _calculate_loss(self, batch, batch_idx, mode, dataset_idx=None):
        """
        Calculate loss for a given batch (similar to ColaMD's _calculate_loss).

        Args:
            batch: Input batch data
            batch_idx: Batch index
            mode: Training mode ("train", "valid", "test")
            dataset_idx: Index of the dataset (for logging)
        Returns:
            torch.Tensor: Computed loss
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            input_values = batch['input_values']
        else:
            input_values = batch

        # Forward pass through wav2vec 2.0
        outputs = self.model(input_values)

        # Get the main loss (contrastive + diversity)
        loss = outputs.loss

        # Create log suffix for dataset-specific logging
        log_suffix = f"{dataset_idx}" if dataset_idx is not None else ""

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

    def training_step(self, batch, batch_idx):
        """
        Training step with multi-dataset support (adapted from ColaMD).
        Expected batch format: (list_of_batches, batch_idx, dataloader_idx)
        """
        # Handle multi-dataset batch format similar to ColaMD
        if isinstance(batch, tuple) and len(batch) == 3:
            # Multi-dataset format: (batch_list, batch_idx, dataloader_idx)
            batch_list, batch_idx, dataloader_idx = batch
            lst = range(len(batch_list))

            # Weighted random selection of dataset (similar to ColaMD)
            selected_dataset = random.choices(lst, weights=self.num_batch, k=1)[0]
            selected_batch = batch_list[selected_dataset]

            # Calculate loss for selected dataset
            loss = self._calculate_loss(selected_batch, batch_idx, "train", selected_dataset)

            # Also log which dataset was selected
            self.log("selected_dataset", float(selected_dataset), batch_size=self.batch_size)

        else:
            # Single dataset format
            loss = self._calculate_loss(batch, batch_idx, "train")

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step with multi-dataset support.
        """
        # Handle multi-dataset batch format
        if isinstance(batch, tuple) and len(batch) == 3:
            batch_list, batch_idx, dataloader_idx = batch
            # For validation, we process all datasets
            total_loss = 0
            for i, dataset_batch in enumerate(batch_list):
                loss = self._calculate_loss(dataset_batch, batch_idx, "valid", i)
                total_loss += loss

            # Log average loss across datasets
            avg_loss = total_loss / len(batch_list)
            self.log("valid_loss_avg", avg_loss, batch_size=self.batch_size)

        else:
            # Single dataset format
            self._calculate_loss(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        """
        Test step with multi-dataset support.
        """
        # Handle multi-dataset batch format
        if isinstance(batch, tuple) and len(batch) == 3:
            batch_list, batch_idx, dataloader_idx = batch
            # For testing, we process all datasets
            for i, dataset_batch in enumerate(batch_list):
                self._calculate_loss(dataset_batch, batch_idx, "test", i)
        else:
            # Single dataset format
            self._calculate_loss(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Configure optimizer similar to ColaMD but with wav2vec 2.0 specific settings.
        """
        # Use AdamW which is better for transformer models
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,  # Adjust based on your training schedule
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def extract_features(self, input_values):
        """
        Extract features from raw audio using the pre-trained wav2vec 2.0 model.
        This method can be used after pre-training to extract representations for downstream tasks.

        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, sequence_length, hidden_size)
        """
        self.model.eval()
        with torch.no_grad():
            # Get the feature extractor and transformer outputs
            outputs = self.model.wav2vec2(input_values)

            # Return the last hidden states (contextualized representations)
            return outputs.last_hidden_state

    def get_quantized_representations(self, input_values):
        """
        Get quantized representations (codebook vectors) from the model.
        Useful for analyzing the learned discrete representations corresponding to cough phases.

        Args:
            input_values (torch.Tensor): Raw audio waveform of shape (batch_size, sequence_length)
        Returns:
            tuple: (quantized_features, codevector_perplexity)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_values)
            return outputs.projected_quantized_states, outputs.codevector_perplexity

    def get_dataset_statistics(self):
        """
        Get statistics about the dataset weights and names.
        Returns:
            dict: Dictionary containing dataset information
        """
        return {
            "dataset_names": self.dataset_names,
            "dataset_weights": self.num_batch,
            "num_datasets": len(self.dataset_names),
            "total_samples": sum(self.hparams.num_batch)
        }