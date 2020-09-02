import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalLoss', 'WeightedFocalLoss']

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        pt = pt.clamp(min=0.00001,max=0.99999)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, gammas=[0, 1], alpha=0.25, reduction='mean'):
        super().__init__()
        self.gammas = gammas
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target, weight = None):
        pt = torch.sigmoid(_input)
        
        loss = torch.zeros_like(_input)
        for gamma in self.gammas:
            loss += - self.alpha * (1 - pt) ** gamma * target * torch.log(pt) - \
                (1 - self.alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt)
        
        loss /= len(self.gammas)
        
        if weight is None:
            weight = torch.ones_like(_input).to(device)
        
        loss *= weight
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss