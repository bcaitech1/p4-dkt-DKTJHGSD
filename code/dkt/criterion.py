
import torch.nn as nn
import torch

def get_criterion(pred, target):

    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)