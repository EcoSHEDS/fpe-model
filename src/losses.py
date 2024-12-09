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
        self.bceloss = nn.BCELoss()

    def forward(self, outputs1, outputs2, targets):
        """Calculate RankNet loss.
        
        Args:
            outputs1: Scores for first items in pairs (batch_size,)
            outputs2: Scores for second items in pairs (batch_size,)
            targets: Target labels as -1, 0, or 1 (batch_size,)
            
        Returns:
            Loss value
        """
        # calculate difference between scores of each image pair
        diff = outputs1 - outputs2  # Shape: (batch_size,)
        
        # calculate probability that sample i should rank higher than sample j
        Pij = torch.sigmoid(diff) 

        # map target labels to probabilities
        target_probs = (targets + 1) / 2  # Map {-1, 0, 1} to {0, 0.5, 1}
        
        # Binary cross entropy between predicted and target probabilities
        return self.bceloss(Pij, target_probs)
