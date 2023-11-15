"""
BANet锛圔oundary-Aware Network锛夋槸涓€绉嶇敤浜庡浘鍍忓垎鍓蹭换鍔＄殑绁炵粡缃戠粶妯″瀷锛屽叾鎹熷け鍑芥暟涓昏�佸寘鎷�涓や釜閮ㄥ垎锛氫氦鍙夌喌鎹熷け鍜岃竟鐣屾崯澶便€�
鍏蜂綋鍦帮紝浜ゅ弶鐔垫崯澶辩敤浜庡害閲忛�勬祴缁撴灉鍜岀湡瀹炴爣绛句箣闂寸殑宸�寮傦紝杈圭晫鎹熷け鍒欑敤浜庨紦鍔辩綉缁滃湪鍒嗗壊杈圭晫澶勪骇鐢熸洿鍔犳竻鏅扮殑棰勬祴缁撴灉銆�
"""
import torch.nn as nn
from .joint_loss import JointLoss
from .dice import DiceLoss
from .soft_ce import SoftCrossEntropyLoss


class BANetLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(BANetLoss, self).__init__()
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
