import lightning as pl
import torch
from pytorch_optimizer import AdamWSN, Ranger
from torchmetrics.classification import Accuracy
from torchmetrics import MeanMetric
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from DisGlioma.src.model.mapvision import VisionModel, VisionConfig
from DisGlioma.src.utils.loss_func import NLLSurvLoss, FocalLoss, RobustCenterLoss


class VisionWrapper(pl.LightningModule):
    def __init__(self, args):
        super(VisionWrapper, self).__init__()
        self.args = args
        self.model = VisionModel(VisionConfig(cross_attn=args.cross_attn))
        self.loss_fn = NLLSurvLoss()
        self.ce_loss = CrossEntropyLoss(ignore_index=-1)
        self.ce_loss = FocalLoss(1, 2, ignore_index=-1)
        self.c_loss = RobustCenterLoss(args.num_cluster, args.center_dim)
        # metric objects for calculating and averaging accuracy across batches

        loss_metric = MeanMetric()
        self.train_loss = deepcopy(loss_metric)
        self.val_loss = deepcopy(loss_metric)
        self.test_loss = deepcopy(loss_metric)

        self.acc = Accuracy(task="multiclass",num_classes=3,ignore_index=-1)
        self.val_acc = Accuracy(task="multiclass",num_classes=3,ignore_index=-1)

    def configure_optimizers(self):
        optimizer = Ranger(self.model.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                  T_max=self.args.max_epochs, 
                                                                  eta_min=self.args.min_lr)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    def forward(self, x):
        return self.model.forward(x)

    def model_step(self, batch, batch_idx):      
        data = batch
        logits, hazards, c_embed = self.forward(data)
        loss = self.loss_fn(hazards, data['label'], data['censor'])
        ce_loss = self.ce_loss(logits, data['cluster'])

        if self.args.cross_attn:
            return loss, logits
        else:
            if self.training:
                c_loss = self.c_loss(c_embed, data['cluster'])
                return loss + ce_loss + 0.0001*c_loss, logits
            else:
                ce_loss = self.ce_loss(logits, data['cluster'])
                return loss + ce_loss, logits

    def training_step(self, batch, batch_idx: int):
        data = batch
        loss, logits = self.model_step(batch, batch_idx)
        # update and log metrics
        self.acc(torch.softmax(logits, dim=-1), data['cluster'])
        self.train_loss(loss)  
        self.log("loss", self.train_loss, sync_dist=True, prog_bar=True)
        self.log("acc", self.acc, sync_dist=True, prog_bar=True)
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
        self.print("train_acc_epoch:{:4f}".format(self.acc.compute().item()),)
        pass

    def validation_step(self, batch, batch_idx: int):
        loss, logits = self.model_step(batch, batch_idx)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(torch.softmax(logits, dim=-1), batch['cluster'])
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
        data = batch
        _, hazards, _ = self.forward(data)
        risk = self.calculate_risk(hazards)
        return risk, data['os'], data['censor']
    
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