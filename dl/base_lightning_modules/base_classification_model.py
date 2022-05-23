from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


# warnings.showwarning = warn_with_traceback


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        self.loss = t.nn.CrossEntropyLoss()
        metrics = MetricCollection(
            [
                Accuracy(num_classes=params.num_classes, average="weighted"),
                Precision(num_classes=params.num_classes, average="weighted"),
                Recall(num_classes=params.num_classes, average="weighted"),
                # F1Score(num_classes=params.num_classes, average="weighted"),
            ]
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self.best_val_accuracy = 0

    def label_to_one_hot(self, labels: t.Tensor, n_classes=8):
        labels = labels.long()
        labels = labels.to(self.device)
        labels = t.eye(n_classes)[labels].to(self.device)
        return labels

    def one_hot_to_label(self, one_hot: t.Tensor, n_classes=8):
        return t.argmax(one_hot, dim=1)

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        pred_y = self.one_hot_to_label(y_pred)
        self.train_metrics.update(pred_y, y)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        metrics = self.val_metrics.compute()
        acc = metrics["Accuracy"]
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_performance", metrics | {"loss": avg_loss})
        self.val_metrics.reset()
        if acc >= self.best_val_accuracy and "save_path" in self.params.__dict__:
            self.best_val_accuracy = acc
            t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "checkpoint.ckpt"),
            )
            print("Saved model as new best.")

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("train_performance", self.train_metrics.compute() | {"loss": avg_loss})
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self(x)
        pred_label = self.one_hot_to_label(pred_y)
        self.val_metrics.update(pred_label, y)
        return {"loss": self.loss(pred_y, y)}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self.one_hot_to_label(self(x))
        self.test_metrics.update(pred_y, y)

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log("test_performance", metrics, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2),  # weight_decay=0.001
        )

        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.params.reduce_lr_on_plateau_patience,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
