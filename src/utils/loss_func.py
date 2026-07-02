import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustCenterLoss(nn.Module):

    def __init__(self, num_classes=3, feat_dim=512):
        super(RobustCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        x: (batch_size, feat_dim)
        labels: (batch_size) containing 0, 1, 2, or -1 (unlabeled)
        """
        # --- 1. ignore label == -1  ---
        labeled_mask = labels != -1
        
        x_labeled = x[labeled_mask]
        labels_labeled = labels[labeled_mask]

        # --- 2. boundary process ---
        # if Batch data.all() == -1 return loss=0.
        if x_labeled.size(0) == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        batch_size = x_labeled.size(0)
        
        # --- 3. compute distance ---
        # distmat = x^2 + c^2 - 2xc
        
        # x_labeled: (B, Dim) -> pow -> sum -> (B, 1) -> expand -> (B, Classes)
        distmat = torch.pow(x_labeled, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
        # addmm: distmat = 1*distmat - 2*(x @ c.t())
        distmat.addmm_(x_labeled, self.centers.t(), beta=1, alpha=-2)

        # --- 4. get distance of ture label ---
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels_labeled = labels_labeled.unsqueeze(1).expand(batch_size, self.num_classes)
        
        # generate mask
        mask = labels_labeled.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        
        # clamp to prevenct (underflow)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check

    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    # c = 1 means censored. Weight 0 in this case 
    censored_loss = -c * (torch.log(s_prev) + torch.log(h_this))
    uncensored_loss = -(1-c) * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss