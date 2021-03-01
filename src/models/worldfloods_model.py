import os
import torch
import wandb
import torchvision
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple

from src.models.utils import losses, metrics
from src.models.utils.model_setup import load_model_weights
from src.models.architectures.baselines import SimpleLinear, SimpleCNN
from src.models.architectures.unets import UNet, UNet_dropout
from src.data.worldfloods.configs import SENTINEL2_NORMALIZATION, COLORS_WORLDFLOODS


class WorldFloodsModel(pl.LightningModule):
    def __init__(self, model_params: dict):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 3)
        self.network = self.configure_architecture(h_params_dict)
        self.weight_per_class = torch.tensor(h_params_dict.get('weight_per_class', [1 for i in range(self.num_class)]))

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)
        
        #label names setup
        self.label_names = h_params_dict.get('label_names', [i for i in range(self.num_class)])
        
        
        ###### IF PRETRAINED WEIGHTS ######
        if model_params.use_pretrained_weights:
            filepath = os.path.join(model_params.path_to_weights, model_params.hyperparameters.model_type, model_params.hyperparameters.model_type  + "_final_weights.pt")
            self.load_pretrained_architecture(filepath)

    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B,  C, W, H), input image
                y (torch.Tensor): (B, num_class, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze()
        logits = self.network(x)
        loss = losses.calc_loss_mask_invalid(logits, y, weight=self.weight_per_class)
        self.log("loss", loss)
        
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits)
            
        return loss
    
    def forward(self, x):
        return self.network(x)

    def log_images(self, x, y, logits):
        mask_data = y.cpu().numpy()
        pred_data = torch.argmax(logits, dim=1).long().cpu().numpy()
        img_data = self.batch_to_unnorm_rgb(x)

        self.logger.experiment.log(
            {"overlay": [self.wb_mask(img, pred, mask) for (img, pred, mask) in zip(img_data, pred_data, mask_data)]})

        self.logger.experiment.log({"image": [wandb.Image(img) for img in img_data]})
        self.logger.experiment.log({"y": [wandb.Image(self.mask_to_rgb(img)) for img in mask_data]})
        self.logger.experiment.log({"pred": [wandb.Image(self.mask_to_rgb(img + 1)) for img in pred_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, num_classes, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask'].squeeze()
        logits = self.network(x)
        
        bce_loss = losses.bce_loss_mask_invalid(logits, y, weight=self.weight_per_class)
        dice_loss = losses.dice_loss_mask_invalid(logits, y)
        self.log('bce_loss', bce_loss)
        self.log('dice_loss', dice_loss)

        pred_categorical = torch.argmax(logits, dim=1).long()

        # cm_batch is (B, num_class, num_class)
        cm_batch = metrics.compute_confusions(y, pred_categorical, num_class=self.num_class,
                                              remove_class_zero=True)

        # TODO log accuracy per class

        # Log IoU per class
        iou_dict = metrics.calculate_iou(cm_batch, self.label_names)
        for k in iou_dict.keys():
            self.log(f"iou {k}", iou_dict[k])
            
        if batch_idx == 0 and self.logger is not None:
            self.log_images(x, y, logits)
        
    def on_val_epoch_end(self):
        self.log('images', self.image_grid)
        self.log('masks', self.mask_grid)
        self.log('preds', self.pred_grid)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "dice_loss"}
    
                                             
    def configure_architecture(self, h_params):
        architecture = h_params.get('model_type', 'linear')
        num_channels = h_params.get('num_channels', 3)
        num_classes = h_params.get('num_classes', 2)
        print(num_channels, num_classes)
                                            
        if architecture == 'unet':
            model = UNet(num_channels, num_classes)

        elif architecture == 'simplecnn':
            model = SimpleCNN(num_channels, num_classes)

        elif architecture == 'linear':
            model = SimpleLinear(num_channels, num_classes)

        elif architecture == 'unet_dropout':
            model = UNet_dropout(num_channels, num_classes)

        else:
            raise Exception(f'No model implemented for model_type: {h_params.model_type}')

        return model
    
    def load_pretrained_architecture(self, filepath):
        load_model_weights(self.network, filepath)

    def batch_to_unnorm_rgb(self, x):
        model_input_npy = x.cpu().numpy()
        norm_facts = SENTINEL2_NORMALIZATION.copy()
        norm_facts = norm_facts[..., np.newaxis, np.newaxis]
        std = norm_facts[np.newaxis,3:0:-1, 1]
        mean = norm_facts[np.newaxis,3:0:-1, 0]
        model_input_rgb_npy = model_input_npy[:, 3:0:-1] * std + mean
        model_input_rgb_npy = np.clip(model_input_rgb_npy / 3000., 0., 1.).transpose((0, 2, 3, 1))
        return model_input_rgb_npy
    
    def wb_mask(self, bg_img, pred_mask, true_mask):
        return wandb.Image(bg_img, masks={
            "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
            "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})
        
    def labels(self):
        return {
            0: "invalid",
            1: "land",
            2: "water",
            3: "cloud"
        }

    def mask_to_rgb(self, mask, values=[0,1,2,3], colors_cmap=COLORS_WORLDFLOODS):
        """
        Given a 2D mask it assign each value of the mask the corresponding color
        :param mask:
        :param values:
        :param colors_cmap:
        :return:
        """
        assert len(values) == len(colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"
        mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
        colores = np.array(np.round(colors_cmap*255), dtype=np.uint8)
        for i, c in enumerate(colores):
            mask_return[mask == values[i], :] = c
        return mask_return
    


