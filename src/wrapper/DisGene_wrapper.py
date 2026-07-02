import lightning as pl
import torch
import torch.nn as nn
from pytorch_optimizer import AdamWSN, Ranger
from ..DisGlioma.disgene import CustomCLIP

from torchmetrics import MeanMetric
from torchmetrics import Accuracy
from copy import deepcopy


class DisGeneWrapper(pl.LightningModule):
    def __init__(self, args):
        super(DisGeneWrapper, self).__init__()
        self.args = args
        self.model = CustomCLIP()
        self.loss_fn = nn.CrossEntropyLoss()
        # metric objects for calculating and averaging accuracy across batches

        loss_metric = MeanMetric()
        self.train_loss = deepcopy(loss_metric)
        self.val_loss = deepcopy(loss_metric)
        self.test_loss = deepcopy(loss_metric)

        self.train_acc = Accuracy('multiclass', num_classes=args.num_cluste)
        self.val_acc = Accuracy('multiclass', num_classes=args.num_cluster)
    def configure_optimizers(self):
        optimizer = Ranger(self.model.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                  T_max=self.args.max_epochs, 
                                                                  eta_min=self.args.min_lr)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    def forward(self, x):
        return self.model.forward(x)

    def model_step(self, batch, batch_idx):      
        x, label, os_time, censor, dis_label = batch
        sim, hazard = self.forward(x)
        loss = self.loss_fn(sim, label)
 
        return loss, sim, label

    def training_step(self, batch, batch_idx: int):
        loss,sim,label = self.model_step(batch, batch_idx)
        # update and log metrics
        self.train_loss(loss)  
        self.train_acc(torch.argmax(sim, dim=-1), label)
        self.log("loss", self.train_loss, sync_dist=True, prog_bar=True)
        self.log("acc", self.train_acc, sync_dist=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        self.print("train_loss_epoch:{:4f}".format(self.train_loss.compute().item()),)
        self.print("train_acc_epoch:{:4f}".format(self.train_acc.compute().item()),)
        pass

    def validation_step(self, batch, batch_idx: int):
        loss,sim,label = self.model_step(batch, batch_idx)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(torch.argmax(sim, dim=-1), label)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        self.print("val_loss_epoch:{:4f}".format(self.val_loss.compute().item()),)
        self.print("val_acc_epoch:{:4f}".format(self.val_acc.compute().item()),)
        pass


    def predict_step(self, batch, batch_idx: int):
        x, label, os_time, censor, dis_label = batch
        sim, hazards = self.forward(x)
        return sim
    
    def calculate_risk(self, h):
        r"""
        Take the logits of the model and calculate the risk for the patient 
        
        Args: 
            - h : torch.Tensor 
        
        Returns:
            - risk : torch.Tensor 
        
        """
        hazards = torch.sigmoid(h)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        
        return risk