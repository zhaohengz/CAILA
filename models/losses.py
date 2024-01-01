import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class MixupClassLoss(nn.Module):
    def __init__(self):
        super(MixupClassLoss, self).__init__()
        self.dim = -1

    def forward(self, pred, target, mixup_pairs, do_mixup, mixup_prob):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # mixup_prob = 1.0 / mixup_pairs.shape[1]
            do_mixup = do_mixup.unsqueeze(1).float()
            true_dist = torch.zeros_like(pred)
            mixup_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), 1)
            mixup_dist.scatter_(1, mixup_pairs, mixup_prob)
            true_dist = true_dist * (1 - do_mixup) + do_mixup * mixup_dist
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class SemiLabelSmoothingLoss(nn.Module):
    def __init__(self, relations, smoothing=0.0, dim=-1):
        super(SemiLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.relations = relations

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred).to(pred.device)
            true_dist.fill_(self.smoothing)
            target_relations = self.relations[target].to(pred.device)
            true_dist = true_dist * target_relations / target_relations.sum(dim=1, keepdim=True)
            for p in true_dist:
                index = p.nonzero(as_tuple=True)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))