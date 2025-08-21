import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCEDice(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, logits, target, mask):

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(probs, target, reduction='none')
        bce = (bce*mask).sum() / (mask.sum()+1e-6)

        inter = (probs*target*mask).sum()
        denom = (probs*mask).sum() + (target*mask).sum() + self.eps
        dice = 1 - (2*inter + self.eps) / denom
        return bce + dice