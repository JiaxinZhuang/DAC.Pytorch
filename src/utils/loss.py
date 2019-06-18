"""Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class modifiedBCELoss(nn.Module):
    def __init__(self, upper_threshold: float, lower_threshold: float, device):
        self.device = device
        self.upper_threshold = torch.tensor(upper_threshold).to(self.device)
        self.lower_threshold = torch.tensor(lower_threshold).to(self.device)

    def forward(self, targets, predicted):
        """Forward
        :param targets:
        :param predicted:
        :return: loss function, ms
        """
        low = torch.where(targets < self.lower_threshold, 1, 0)
        up = torch.where(targets > self.upper_threshold, 1, 0)
        weigth = low + up
        threshold = weigth / 2
        num_selected = torch.sum(weigth)
        selected_predicted = torch.where(predicted > threshold, 1, 0)
        loss = torch.sum(weigth * F.binary_cross_entropy(selected_predicted, targets, reduction="sum")/num_selected)
        return loss
