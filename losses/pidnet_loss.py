import torch.nn as nn
from .joint_loss import JointLoss
from .dice import DiceLoss
from .soft_ce import SoftCrossEntropyLoss

class PIDNetLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(PIDNetLoss, self).__init__()
        #self.joint_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        #                            DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, y_pred, y_true):
        if self.training:
            #joint_loss = self.joint_loss(y_pred, y_true)
            joint_loss = self.ce_loss(y_pred,y_true)
        else:
            #joint_loss = self.joint_loss(y_pred, y_true)
            joint_loss = self.ce_loss(y_pred,y_true)
        return joint_loss

