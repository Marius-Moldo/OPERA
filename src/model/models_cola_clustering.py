import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
import numpy as np
from src.model.htsat.htsat import HTSATWrapper
import random


class Encoder(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)

        # Try alternative approach to loading EfficientNet
        try:
            self.efficientnet = EfficientNet.from_name(
                "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
            )
        except (AssertionError, KeyError) as e:
            print(f"Warning: Failed to load pretrained weights with include_top=False: {e}")
            print("Loading with include_top=True and removing classifier manually...")

            # Load with classifier and remove it manually
            self.efficientnet = EfficientNet.from_pretrained(
                "efficientnet-b0", include_top=True, drop_connect_rate=drop_connect_rate
            )
            # Remove the final classification layer
            self.efficientnet._fc = torch.nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        # Debug: print tensor shape
        # print(f"EfficientNet output shape: {x.shape}")

        # Handle different output shapes more robustly
        if len(x.shape) == 4 and x.shape[2] == 1 and x.shape[3] == 1:
            # Expected case: [batch, channels, 1, 1]
            y = x.squeeze(3).squeeze(2)
        elif len(x.shape) == 4:
            # If spatial dimensions are not 1x1, use adaptive pooling
            y = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2)
        elif len(x.shape) == 2:
            # Already flattened
            y = x
        else:
            # Flatten to 2D
            y = x.view(x.size(0), -1)

        return y


class EncoderHTSAT(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(EncoderHTSAT, self).__init__()
        self.encoder = HTSATWrapper()
        self.out_emb = 768

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.encoder(x)

        return y


class Cola(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet",
                 max_len=251, out_emb=2048,
                 num_clusters=3,  # Number of clusters (e.g., for cough phases)
                 cluster_temperature=0.1  # Temperature for sharpening cluster assignments
                 ):
        super().__init__()
        self.save_hyperparameters()

        # --- Existing initializations ---
        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea: self.dim_hidden = self.dim_fea
        self.encoder_model = encoder

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.dim_out)

        # NOTE: self.linear is now part of the projector, not for the final contrastive comparison
        self.projector = torch.nn.Linear(self.dim_out, self.dim_out, bias=False)

        # --- Online Clustering Additions ---
        self.num_clusters = num_clusters
        self.cluster_temperature = cluster_temperature

        # These are the learnable cluster centroids
        self.cluster_centroids = torch.nn.Parameter(torch.randn(self.num_clusters, self.dim_out))

    def forward(self, x):
        x1, x2 = x

        # --- Pass both views through the encoder and projector ---
        # We need the features *before* the final projection for clustering

        # View 1
        h1_fea = self.encoder(x1)
        if self.middle_enabled:
            h1_fea = self.do(self.middle(h1_fea))
        else:
            h1_fea = self.do(h1_fea)
        h1_proj = self.do(self.g(h1_fea))
        h1_norm = self.do(torch.tanh(self.layer_norm(h1_proj)))

        # View 2
        h2_fea = self.encoder(x2)
        if self.middle_enabled:
            h2_fea = self.do(self.middle(h2_fea))
        else:
            h2_fea = self.do(h2_fea)
        h2_proj = self.do(self.g(h2_fea))
        h2_norm = self.do(torch.tanh(self.layer_norm(h2_proj)))

        # Final projection for loss calculation
        z1 = self.projector(h1_norm)
        z2 = self.projector(h2_norm)

        # Return both the final projections (z) and the normalized features for clustering (h)
        return z1, z2, h1_norm, h2_norm

    # A helper function for the swapped prediction loss
    def _cluster_loss(self, z, h):
        # Normalize features and centroids for stable training
        z_norm = F.normalize(z, dim=1)
        centroids_norm = F.normalize(self.cluster_centroids, dim=1)

        # Calculate similarity between features and centroids
        scores = torch.mm(z_norm, centroids_norm.t())  # Shape: [batch_size, num_clusters]

        # Get cluster assignments (targets) from the *other* view's features (h)
        # We use .detach() because these assignments are treated as targets, not part of the gradient path.
        with torch.no_grad():
            targets = torch.argmax(torch.mm(F.normalize(h, dim=1), centroids_norm.t()), dim=1)

        # Calculate the cross-entropy loss
        # The model predicts cluster assignments for view 'z' using the assignments from view 'h' as ground truth.
        loss = F.cross_entropy(scores / self.cluster_temperature, targets)

        # Calculate accuracy for logging
        _, predicted = torch.max(scores, 1)
        acc = (predicted == targets).double().mean()

        return loss, acc

    def training_step(self, x, batch_idx):
        # The forward pass now returns final projections (z) and features for clustering (h)
        z1, z2, h1, h2 = self(x)

        # --- MODIFIED LOSS CALCULATION ---
        # This is the "swapped" prediction task.
        # Loss 1: Predict cluster assignments for view 1 using assignments from view 2 as targets.
        loss1, acc1 = self._cluster_loss(z1, h2)

        # Loss 2: Predict cluster assignments for view 2 using assignments from view 1 as targets.
        loss2, acc2 = self._cluster_loss(z2, h1)

        # Total loss is the average of the two
        loss = (loss1 + loss2) / 2
        acc = (acc1 + acc2) / 2

        self.log("train_loss", loss)
        self.log("train_cluster_acc", acc)

        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        # Validation step uses the same logic
        z1, z2, h1, h2 = self(x)

        loss1, acc1 = self._cluster_loss(z1, h2)
        loss2, acc2 = self._cluster_loss(z2, h1)

        loss = (loss1 + loss2) / 2
        acc = (acc1 + acc2) / 2

        self.log("valid_loss", loss, sync_dist=True)
        self.log("valid_cluster_acc", acc, sync_dist=True)

    def test_step(self, x, batch_idx):
        z1, z2, h1, h2 = self(x)

        loss1, acc1 = self._cluster_loss(z1, h2)
        loss2, acc2 = self._cluster_loss(z2, h1)

        loss = (loss1 + loss2) / 2
        acc = (acc1 + acc2) / 2

        self.log("test_loss", loss)
        self.log("test_cluster_acc", acc)

    def extract_feature(self, x, dim=1280):
        if self.encoder_model == "vit":
            return self.extract_feature_vit(x, dim)
        x = self.encoder(x)
        if dim == self.dim_fea:
            return x
        if self.middle_enabled:
            x = self.middle(x)
        if dim == self.dim_hidden:
            return x
        x = self.g(x)
        if dim == self.dim_out:
            return x
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4e-4)


class ColaMD(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet",
                 batch_size=128, num_batch=[258.0, 288, 4, 51, 75, 146, 138], out_emb=2048, max_len=251,
                 num_clusters=3,  # Number of clusters (e.g., for cough phases)
                 cluster_temperature=0.1  # Temperature for sharpening cluster assignments
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea: self.dim_hidden = self.dim_fea
        self.encoder_model = encoder

        print(num_batch)
        self.num_batch = [b / np.sum(num_batch) for b in num_batch]
        print(self.num_batch)

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=dim_out)

        # NOTE: Renamed to projector for consistency with Cola
        self.projector = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.batch_size = batch_size

        # --- Online Clustering Additions ---
        self.num_clusters = num_clusters
        self.cluster_temperature = cluster_temperature

        # These are the learnable cluster centroids
        self.cluster_centroids = torch.nn.Parameter(torch.randn(self.num_clusters, self.dim_out))

    def forward(self, x):
        x1, x2 = x

        # --- Pass both views through the encoder and projector ---
        # We need the features *before* the final projection for clustering

        # View 1
        h1_fea = self.encoder(x1)
        if self.middle_enabled:
            h1_fea = self.do(self.middle(h1_fea))
        else:
            h1_fea = self.do(h1_fea)
        h1_proj = self.do(self.g(h1_fea))
        h1_norm = self.do(torch.tanh(self.layer_norm(h1_proj)))

        # View 2
        h2_fea = self.encoder(x2)
        if self.middle_enabled:
            h2_fea = self.do(self.middle(h2_fea))
        else:
            h2_fea = self.do(h2_fea)
        h2_proj = self.do(self.g(h2_fea))
        h2_norm = self.do(torch.tanh(self.layer_norm(h2_proj)))

        # Final projection for loss calculation
        z1 = self.projector(h1_norm)
        z2 = self.projector(h2_norm)

        # Return both the final projections (z) and the normalized features for clustering (h)
        return z1, z2, h1_norm, h2_norm

    # A helper function for the swapped prediction loss
    def _cluster_loss(self, z, h):
        # Normalize features and centroids for stable training
        z_norm = F.normalize(z, dim=1)
        centroids_norm = F.normalize(self.cluster_centroids, dim=1)

        # Calculate similarity between features and centroids
        scores = torch.mm(z_norm, centroids_norm.t())  # Shape: [batch_size, num_clusters]

        # Get cluster assignments (targets) from the *other* view's features (h)
        # We use .detach() because these assignments are treated as targets, not part of the gradient path.
        with torch.no_grad():
            targets = torch.argmax(torch.mm(F.normalize(h, dim=1), centroids_norm.t()), dim=1)

        # Calculate the cross-entropy loss
        # The model predicts cluster assignments for view 'z' using the assignments from view 'h' as ground truth.
        loss = F.cross_entropy(scores / self.cluster_temperature, targets)

        # Calculate accuracy for logging
        _, predicted = torch.max(scores, 1)
        acc = (predicted == targets).double().mean()

        return loss, acc

    def extract_feature(self, x, dim=1280):
        x = self.encoder(x)
        if dim == self.dim_fea:
            return x
        if self.middle_enabled:
            x = self.middle(x)
        if dim == self.dim_hidden:
            return x
        x = self.g(x)
        if dim == self.dim_out:
            return x
        raise NotImplementedError

    def _calculate_loss(self, x, batch_idx, mode):
        # The forward pass now returns final projections (z) and features for clustering (h)
        z1, z2, h1, h2 = self(x)

        # --- MODIFIED LOSS CALCULATION ---
        # This is the "swapped" prediction task.
        # Loss 1: Predict cluster assignments for view 1 using assignments from view 2 as targets.
        loss1, acc1 = self._cluster_loss(z1, h2)

        # Loss 2: Predict cluster assignments for view 2 using assignments from view 1 as targets.
        loss2, acc2 = self._cluster_loss(z2, h1)

        # Total loss is the average of the two
        loss = (loss1 + loss2) / 2
        acc = (acc1 + acc2) / 2

        self.log("{}_loss".format(mode), loss, batch_size=self.batch_size)
        self.log("{}_cluster_acc".format(mode), acc, batch_size=self.batch_size)
        return loss

    def training_step(self, x, batch_idx):
        """
        covidbreath Length of Training, Validation, Testing: 258 29 29
        covidcough Length of Training, Validation, Testing: 288 32 32
        icbhi Length of Training, Validation, Testing: 4 1 1
        coughvid Length of Training, Validation, Testing: 51 6 6
        hf_lung Length of Training, Validation, Testing: 75 9 9
        covidUKexhalation Length of Training, Validation, Testing: 146 17 17
        covidUKcough Length of Training, Validation, Testing: 138 16 16
        """

        batch, batch_idx, dataloader_idx = x
        lst = range(len(batch))

        s = random.choices(lst, weights=self.num_batch, k=1)[0]
        loss = self._calculate_loss(batch[s], batch_idx, "train" + str(s))
        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        batch, batch_idx, dataloader_idx = x

        self._calculate_loss(batch, batch_idx, "valid")

    def test_step(self, x, batch_idx):
        batch, batch_idx, dataloader_idx = x
        self._calculate_loss(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def weights_init(network):
    for m in network:
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()