import numpy as np

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat



class Accuracy:

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        self.correct += (preds == targets).sum()
        self.total += len(preds)

    def result(self):
        if (self.total <= 0):
            return 0
        return self.correct/self.total


class SeqAccuracy:

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):

        seq_len = preds.shape[1]

        self.correct += (((preds == targets).sum(1)) == seq_len).sum()
        self.total += preds.shape[0]

    def result(self):
        if (self.total <= 0):
            return 0
        return self.correct/self.total


class MSE:

    def __init__(self):
        self.diff = 0
        self.total = 0

    def reset(self):
        self.diff = 0
        self.total = 0

    def update(self, preds, targets):
        self.diff += np.square(preds - targets).sum()
        self.total += len(preds)

    def result(self):
        if (self.total <= 0):
            return 0
        return self.diff/self.total






########################## <OBNBenchmark> ##########################
# https://github.com/krishnanlab/obnbench/blob/main/obnbench/metrics.py
class AUROC(torchmetrics.classification.MultilabelAUROC):
    ...

class AP(torchmetrics.classification.MultilabelAveragePrecision):
    ...

class APOP(AP):

    def __init__(self, task="multilabel", *args, average="macro", **kwargs):
        if task != "multilabel":
            raise NotImplementedError(
                "AveragePrecisionOverPrior is ony implemented for "
                "multilabel task for now.",
            )
        self.__average = average
        super().__init__(*args, average="none", **kwargs)

    def compute(self) -> torch.Tensor:
        scores = super().compute()

        # XXX: Does not consider negative selection currently.
        target = dim_zero_cat(self.target)
        prior = target.sum(0).clamp(1) / target.shape[0]
        scores = torch.log2(scores / prior)

        if self.__average == "macro":
            return scores.mean()
        elif self.__average == "none":
            return scores
        else:
            raise ValueError(f"Unknown averaging option {self.__average!r}")
########################## </OBNBenchmark> ##########################