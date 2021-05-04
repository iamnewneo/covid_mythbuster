import torch
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class ComythRationaleModel(pl.LightningModule):
    def __init__(self, model_name, lr_base, lr_linear, *args, **kwargs):
        self.lr_base = lr_base
        self.lr_linear = lr_linear
        super().__init__(*args, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[
                {"params": self.model.roberta.parameters(), "lr": self.lr_base},
                {"params": self.model.classifier.parameters(), "lr": self.lr_linear},
            ]
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], labels=batch["y"].unsqueeze(-1))
        loss = outputs.loss
        logits = outputs.logits
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], labels=batch["y"].unsqueeze(-1))
        loss = outputs.loss
        logits = outputs.logits
        y_pred = logits.argmax(dim=1)
        y_true = batch["y"]
        result = {"loss": loss, "y_pred": y_pred, "y_true": y_true, "y_logit": logits}
        return result

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        self.temp_train_loss = avg_train_loss

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            avg_val_loss = torch.tensor(
                [x["loss"].mean() for x in val_step_outputs]
            ).mean()
            preds = torch.cat([x["y_pred"] for x in val_step_outputs], axis=0)
            targets = torch.cat([x["y_true"] for x in val_step_outputs], axis=0)
            self.log("val_loss", avg_val_loss, prog_bar=True)
            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()
            result = {
                "macro_f1": f1_score(
                    targets_cpu, preds_cpu, zero_division=0, average="macro"
                ),
                "accuracy": accuracy_score(targets_cpu, preds_cpu),
            }
            text = (
                f"Epoch: {self.current_epoch} Train Loss: {self.temp_train_loss:.2f}"
                f" Val Loss: {avg_val_loss:.2f} Macro F1: {result['macro_f1']:.2f}"
                f" Accuracy: {result['accuracy']:.2f}"
            )
            print(text)


class ComythLabelModel(pl.LightningModule):
    def __init__(self, model_name, lr_base, lr_linear, *args, **kwargs):
        self.lr_base = lr_base
        self.lr_linear = lr_linear
        super().__init__(*args, **kwargs)
        config = AutoConfig.from_pretrained(model_name, num_labels=3)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[
                {"params": self.model.roberta.parameters(), "lr": self.lr_base},
                {"params": self.model.classifier.parameters(), "lr": self.lr_linear},
            ]
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], labels=batch["y"].unsqueeze(-1))
        loss = outputs.loss
        logits = outputs.logits
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], labels=batch["y"].unsqueeze(-1))
        loss = outputs.loss
        logits = outputs.logits
        y_pred = logits.argmax(dim=1)
        y_true = batch["y"]
        result = {"loss": loss, "y_pred": y_pred, "y_true": y_true, "y_logit": logits}
        return result

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        self.temp_train_loss = avg_train_loss

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            avg_val_loss = torch.tensor(
                [x["loss"].mean() for x in val_step_outputs]
            ).mean()    
            preds = torch.cat([x["y_pred"] for x in val_step_outputs], axis=0)
            targets = torch.cat([x["y_true"] for x in val_step_outputs], axis=0)
            self.log("val_loss", avg_val_loss, prog_bar=True)
            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()
            result = {
                "macro_f1": f1_score(
                    targets_cpu, preds_cpu, zero_division=0, average="macro"
                ),
                # "f1": f1_score(targets, preds, zero_division=0, average=None),
                # "precision": precision_score(
                #     targets, preds, zero_division=0, average=None
                # ),
                # "recall": recall_score(targets, preds, zero_division=0, average=None),
                "accuracy": accuracy_score(targets_cpu, preds_cpu),
            }
            # text = (
            #     f"\nEpoch: {self.current_epoch} Precision: {result['precision']:.2f}"
            #     f" Recall: {result['recall']:.2f} F1: {result['f1']:.2f}"
            # )
            text = (
                f"Epoch: {self.current_epoch} Train Loss: {self.temp_train_loss:.2f}"
                f" Val Loss: {avg_val_loss:.2f} Macro F1: {result['macro_f1']:.2f}"
                f" Accuracy: {result['accuracy']:.2f}"
            )
            print(text)
