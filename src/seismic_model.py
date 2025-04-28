import pytorch_lightning as pl
import torch
import torch.nn as nn


class SeismicMultilabelModel(pl.LightningModule):
    """
    PyTorch Lightning model for multilabel seismic event classification.

    Args:
        input_dim (int): Number of input features.
        hidden_units (tuple): Sizes of hidden layers.
        dropout_rate (float): Dropout probability between layers.
        lr (float): Learning rate for the optimizer.
        l2 (float): Weight decay (L2 regularization).
        num_classes (int): Number of output labels.
        **kwargs: Additional hyperparameters (ignored).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units=(128,),
        dropout_rate: float = 0.0,
        lr: float = 1e-3,
        l2: float = 1e-4,
        num_classes: int = 3,
        **kwargs,
    ):
        super().__init__()
        # Save hyperparameters to self.hparams
        self.save_hyperparameters()

        # Build an MLP: [input] -> hidden layers -> output
        layers = []
        prev_dim = self.hparams.input_dim
        for h in self.hparams.hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if self.hparams.dropout_rate > 0:
                layers.append(nn.Dropout(self.hparams.dropout_rate))
            prev_dim = h
        # Final output layer
        layers.append(nn.Linear(prev_dim, self.hparams.num_classes))
        self.model = nn.Sequential(*layers)

        # Use BCE with logits for multilabel
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2
        )
        return optimizer
