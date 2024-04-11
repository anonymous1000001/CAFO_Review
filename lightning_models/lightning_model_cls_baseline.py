import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.shufflenet import ShuffleNet
from model.resnet9 import ResNet9
from lightning_models.viz_confusion import make_confusion_matrix


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.task.task_name in [
            "gilon_activity",
            "microsoft_activity",
            "microsoft_activity_new",
            "squid_game",
            "fingergesture",
            "SpokenArabicDigits",
        ]
        self.lr = cfg.task.optimizer.lr
        self.model_name = cfg.model.model_name
        if self.model_name == "shufflenet":
            self.model = ShuffleNet(cfg)
        elif self.model_name == "resnet9":
            self.model = ResNet9(cfg)
        elif self.model_name == "simplevit":
            from model.simpleVIT import SimpleVIT

            self.model = SimpleVIT(cfg)
        elif self.model_name == "mlpmixer":
            from model.mixer import MlpMixer

            self.model = MlpMixer(cfg)
        else:
            raise ValueError(f"Unknown model {self.model_name}")
        self.num_params = sum([p.numel() for p in self.model.parameters()])
        self.log("num_params", self.num_params)

        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        if batch_idx == 0:
            print(y_true[:5])
        y_pred = self.model(feature)
        loss = self.loss(y_pred.squeeze(), y_true)
        self.log("loss", loss.item(), on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred = self.model(feature)
        loss = self.loss(y_pred.squeeze(), y_true)
        return {"val_loss": loss, "y_pred": y_pred, "label": y_true}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        # Collect results from all validation batches
        y_preds = torch.cat([torch.argmax(x["y_pred"], dim=1) for x in outputs]).cpu()
        labels = torch.cat([x["label"] for x in outputs]).cpu()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        accuracy = accuracy_score(y_preds, labels)
        print(f"val_loss: {val_loss:.4f}, accuracy: {accuracy:.4f}")

        self.log_dict({"val_loss": val_loss, "val_accuracy": accuracy})

    def test_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred = self.model(feature)
        loss = self.loss(y_pred.squeeze(), y_true)

        return {"y_pred": y_pred, "y_true": y_true, "test_loss": loss}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).cpu().mean()
        y_preds = torch.cat([torch.argmax(x["y_pred"], 1) for x in outputs]).cpu()
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu()
        test_accuracy = accuracy_score(y_true, y_preds)
        if self.cfg.save_weights:
            self._save_weight()
        # Save the test results in the output directory
        test_label = pd.read_csv(f"{self.cfg.save_output_path}/test_label.csv")
        test_label["test_loss"] = test_loss.item()
        test_label["test_accuracy"] = test_accuracy
        test_label["seed"] = self.cfg.seed
        test_label["exp_num"] = self.cfg.exp_num
        test_label["y_pred"] = y_preds.numpy()
        test_label["y_true"] = y_true.numpy()
        test_label["cv_num"] = self.cfg.task.validation_cv_num
        test_label["model_name"] = self.cfg.model.model_name
        test_label["task_name"] = self.cfg.task.task_name
        test_label["project_name"] = self.cfg.logger.name
        test_label.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv", index=False
        )

        activity_cf_mat = confusion_matrix(y_true, y_preds)
        make_confusion_matrix(
            activity_cf_mat, f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_confusion_matrix.png"
        )

        self.logger.experiment["test_labelfile"].upload(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv"
        )
        self.logger.experiment["confusion_matrix"].upload(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_confusion_matrix.png"
        )
        self.log_dict({"test_accuracy": accuracy_score(y_true, y_preds)})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.cfg.task.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.task.scheduler_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _save_weight(self):
        with open(
            f"{self.cfg.save_output_path}/{self.cfg.model.model_name}_cv{self.cfg.task.validation_cv_num}_model.pt",
            "wb",
        ) as f:
            torch.save(self.model, f)

        print(f"Saved model to {self.cfg.model.model_name}_cv{self.cfg.task.validation_cv_num}_model.pt")
