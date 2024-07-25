import pickle

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.modules import ResNetRankNet


class RankNetModule(pl.LightningModule):
    def __init__(
        self,
        model_args,
        criterion,
        lr,
        validation_metrics=None,
        freeze_layers=None,
        unfreeze_after=0,
    ):
        super().__init__()
        self.model = ResNetRankNet(**model_args)
        self.criterion = criterion
        self.lr = lr
        self.validation_metrics = (
            validation_metrics if validation_metrics else [criterion]
        )
        self.freeze_layers = freeze_layers if freeze_layers else []
        self.unfreeze_after = unfreeze_after

        # If unfreeze_after > 0, freeze the specified layers
        if self.unfreeze_after > 0:
            for name, module in self.model.named_modules():
                if name in self.freeze_layers:
                    print(f"Freezing layer: {name}")
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y1_hat, y2_hat = self.model(x1, x2)
        loss = self.criterion(y1_hat, y2_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y1_hat, y2_hat = self.model(x1, x2)
        loss = self.criterion(y1_hat, y2_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # Compute all validation metrics
        for metric in self.validation_metrics:
            if metric is self.criterion:
                metric_val = loss
            else:
                metric_val = metric(y1_hat, y2_hat, y)
            self.log(
                f"val_{metric.__class__.__name__}",
                metric_val,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        y1_hat, y2_hat = self.model(x1, x2)
        loss = self.criterion(y1_hat, y2_hat, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # Compute all validation metrics
        for metric in self.validation_metrics:
            if metric is self.criterion:
                metric_val = loss
            else:
                metric_val = metric(y1_hat, y2_hat, y)
            self.log(
                f"test_{metric.__class__.__name__}",
                metric_val,
                on_epoch=True,
                prog_bar=True,
            )
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.model.forward_single(x)
        return y_hat

    def configure_optimizers(self):
        # Option 1: Keep initially trainable parameters and unfrozen parameters separate
        # initially_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = SGD(initially_trainable_params, lr=self.lr, momentum=0.9)
        
        # Option 2: Keep all parameters in the same optimizer
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        # NOTE: ReduceLROnPlateau might work only with a single learning rate across all parameter groups
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=1, factor=0.5
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def on_validation_epoch_end(self):
        # If we've trained for unfreeze_after epochs and unfreeze_after > 0, unfreeze the specified layers
        print(f'Current epoch: {self.current_epoch + 1}, unfreeze_after: {self.unfreeze_after}')
        if self.unfreeze_after > 0 and self.current_epoch + 1 == self.unfreeze_after:
            unfrozen_params = []
            for name, module in self.model.named_modules():
                if name in self.freeze_layers:
                    for param in module.parameters():
                        param.requires_grad = True
                        unfrozen_params.append(param)

            # # Option 1: If we added only the initially trainable parameters to the optimizer,
            # # add the newly unfrozen parameters to the optimizer now
            # self.optimizer.add_param_group(
            #     {"params": unfrozen_params, "lr": self.lr * 0.1}
            # )

            # Option 2: If we kept all parameters in the same optimizer, don't do anything
            

class LossLoggingCallback(pl.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.losses = {"train_loss": [], "val_loss": [], "test_loss": None}

    def on_train_epoch_end(self, trainer, pl_module):
        if 'train_loss' in trainer.logged_metrics:
            train_loss = trainer.logged_metrics['train_loss'].item()
            self.losses['train_loss'].append((trainer.current_epoch, train_loss))        
        self.save_losses()

    def on_validation_epoch_end(self, trainer, pl_module):
        if 'val_loss' in trainer.logged_metrics:
            val_loss = trainer.logged_metrics['val_loss'].item()
            self.losses['val_loss'].append((trainer.current_epoch, val_loss))
        self.save_losses()
    
    def on_test_epoch_end(self, trainer, pl_module):
        if 'test_loss' in trainer.logged_metrics:
            test_loss = trainer.logged_metrics['test_loss'].item()
            self.losses['test_loss'] = test_loss
        self.save_losses()

    def save_losses(self):
        print(f"Saving losses to {self.filepath}")
        with open(self.filepath, "wb") as f:
            pickle.dump(self.losses, f)
