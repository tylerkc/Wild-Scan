# AAI-590 Group 9
# Custom Loss Functions to be used during initial training and re-training
# to be updated later

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyMarginLoss(nn.Module):
    """
    Custom loss: CrossEntropy + Margin (logit or probability).
    Args:
        margin_lambda (float): strength of the margin penalty
        margin_type (str): "logits" or "probs" - margin on logits or softmax probabilities
    """
    def __init__(self, reduction = 'mean', margin_lambda=0.1, margin_type="logits"): # defaults
        super().__init__()
        self.reduction = reduction
        self.margin_lambda = margin_lambda
        assert margin_type in ("logits", "probs"), "Invalid margin_type"
        self.margin_type = margin_type

    def forward(self, logits, targets):
        
        # Standard CrossEntropyLoss (on logits)
        ce_loss = F.cross_entropy(logits, targets, reduction = self.reduction)
        
        # Margin computation (logits or softmax probabilities)
        if self.margin_type == "logits":
            sorted_logits, _ = torch.sort(logits, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        else:  # "softmax probs"
            probs = F.softmax(logits, dim=1)
            sorted_probs, _ = torch.sort(probs, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        # Penalty: encourage larger margins
        if self.reduction == 'mean':
            margin_penalty = -torch.mean(margin)
        else:
            margin_penalty = -margin

        # combine ce_loss and effect of margin maximization
        loss = ce_loss + self.margin_lambda * margin_penalty
        
        return loss, ce_loss

    def update_params(self, reduction = None, margin_lambda=None, margin_type=None):
        """Dynamically update loss hyperparameters."""
        if reduction is not None:
            assert reduction in ("mean", "none"), "Invalid reduction type"
            self.reduction = reduction
        if margin_lambda is not None:
            self.margin_lambda = margin_lambda
        if margin_type is not None:
            assert margin_type in ("logits", "probs"), "Invalid margin_type"
            self.margin_type = margin_type