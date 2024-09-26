import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.functional import auroc
from torchvision import models


class MammoNet(pl.LightningModule):
    def __init__(
        self, num_classes, backbone="resnet18", learning_rate=0.0001, checkpoint=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = learning_rate
        self.backbone = backbone

        # Default model is a ResNet-18 pre-trained on ImageNet
        if self.backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.backbone == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        if checkpoint is not None:
            print(self.model.load_state_dict(state_dict=checkpoint, strict=False))

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def process_batch(self, batch):
        img, lab = batch["image"], batch["label"]
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = F.cross_entropy(out, lab)
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
            num_classes=self.num_classes,
            average="macro",
            task="multiclass",
        )
        self.log("train_auc", auc)
        self.train_preds = []
        self.train_trgts = []

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
            num_classes=self.num_classes,
            average="macro",
            task="multiclass",
        )
        self.log("val_auc", auc)
        self.val_preds = []
        self.val_trgts = []

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_trgts = []
        self.test_study_ids = []
        self.test_image_ids = []

    def test_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.log("test_loss", loss, batch_size=lab.shape[0])
        self.test_preds.append(prd.detach().cpu())
        self.test_trgts.append(lab.detach().cpu())
        self.test_study_ids.append(batch["study_id"])
        self.test_image_ids.append(batch["image_id"])

    def on_test_epoch_end(self):
        self.test_preds = torch.cat(self.test_preds, dim=0)
        self.test_trgts = torch.cat(self.test_trgts, dim=0)
        auc = auroc(
            self.test_preds,
            self.test_trgts,
            num_classes=self.num_classes,
            average="macro",
            task="multiclass",
        )
        self.log("test_auc", auc)
