import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.functional import auroc
from sklearn.metrics import balanced_accuracy_score
import numpy as np

MARKER_NAMES = [
    "circle marker",
    "triangle marker",
    "breast implant",
    "devices",
    "compression",
]


class Multilabel_ArtifactDetector(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = learning_rate
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch):
        img, lab = batch["image"], batch["label"]
        out = self.model(img)
        prd = torch.sigmoid(out)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, lab.float())
        return loss, prd, lab

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_trgts = []

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log("train_loss", loss, batch_size=lab.shape[0])
        self.train_preds.append(prd.detach().cpu())
        self.train_trgts.append(lab.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        self.train_preds = torch.cat(self.train_preds, dim=0)
        self.train_trgts = torch.cat(self.train_trgts, dim=0)
        auc = auroc(
            self.train_preds,
            self.train_trgts,
            average="macro",
            task="multilabel",
            num_labels=self.num_classes,
        )
        self.log("train_auc", auc)

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_trgts = []

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log("val_loss", loss, batch_size=lab.shape[0])
        self.val_preds.append(prd.detach().cpu())
        self.val_trgts.append(lab.detach().cpu())

    def on_validation_epoch_end(self):
        self.val_preds = torch.cat(self.val_preds, dim=0)
        self.val_trgts = torch.cat(self.val_trgts, dim=0)
        auc = auroc(
            self.val_preds,
            self.val_trgts,
            average="macro",
            task="multilabel",
            num_labels=self.num_classes,
        )
        self.log("val_auc", auc)
        all_bal_acc = [
            balanced_accuracy_score(self.val_preds[:, i] > 0.5, self.val_trgts[:, i])
            for i in range(5)
        ]
        [self.log(f"val_bal_acc_{i}", all_bal_acc[i]) for i in range(5)]
        self.log("val_bal_acc", np.asarray(all_bal_acc).mean())

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_trgts = []
        self.test_image_ids = []

    def test_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log("test_loss", loss, batch_size=lab.shape[0])
        self.test_preds.append(prd.detach().cpu())
        self.test_trgts.append(lab.detach().cpu())
        self.test_image_ids.append(batch["image_id"])

    def on_test_epoch_end(self):
        self.test_preds = torch.cat(self.test_preds, dim=0)
        self.test_trgts = torch.cat(self.test_trgts, dim=0)
        auc = auroc(
            self.test_preds,
            self.test_trgts,
            average="macro",
            task="multilabel",
            num_labels=self.num_classes,
        )
        self.log("test_auc", auc)
