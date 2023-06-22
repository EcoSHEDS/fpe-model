import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        output = self.loss(input, target)
        return output


class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, inputs_i, inputs_j, targets):
        oij = inputs_i - inputs_j
        Pij = torch.sigmoid(oij)
        target_probs = 0.5 * (targets + 1)
        bceloss = nn.BCELoss()
        loss = bceloss(Pij, target_probs)
        return loss
