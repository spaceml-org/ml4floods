from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch

from src.models.utils import losses, metrics


class WorldFloodsModel(pl.LightningModule):
    def __init__(self, 
                 network_architecture: torch.nn.Module, 
                 num_class:int,
                 weight_per_class: Optional[List[float]]=None, 
                 lr:float=1e-4, 
                 lr_decay:float=.5,
                 lr_patience:int=2):
        
        super().__init__()
        self.num_class = num_class
        self.network = network_architecture
        self.weight_per_class = torch.tensor(weight_per_class)

        # learning rate params
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, num_class, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid(logits, y, weight=self.weight_per_class)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, num_classes, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch
        logits = self.network(x)

        bce_loss = losses.cross_entropy_loss_mask_invalid(logits, y, weight=self.weight_per_class)
        dice_loss = losses.dice_loss_mask_invalid(logits, y)
        self.log('bce_loss', bce_loss)
        self.log('dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class, remove_class_zero=True)

        # TODO log accuracy per class

        # TODO log IoU per class

        # TODO plot some input output images -> perhaps this should go to validation_epoch_end?


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "dice_loss"}





