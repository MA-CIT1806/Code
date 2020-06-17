import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from scipy.sparse.linalg.eigen.arpack import eigsh
from pytorch_metric_learning import losses, miners


def chebyshev_polynomials(base, k, debug=False, kind=1):
    """Calculate Chebyshev polynomials up to order k. Return a list of matrices (tuple representation)."""
    if debug:
        print("Calculating Chebyshev polynomials up to order {}...".format(k))        

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, base):
        mx = base.clone()
        return (2 * mx * t_k_minus_one) - t_k_minus_two

    t_k = list()

    x0 = base.new_ones(base.shape)
    t_k.append(x0)
    t_k.append(kind * base)             

    for _ in range(2, k):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], base))

    return t_k


class LocalSaveHandler():
    "Keeps an updated checkpoint of the best model internally. Checkpoint is not written to disk."

    def __init__(self, pipeline):
        self.pipeline = pipeline

        self.checkpoints = {}

    def __call__(self, checkpoint, filename):
        self.checkpoints[filename] = checkpoint

        epoch = re.findall("_[0-9]+_", filename)[0].replace("_", "")

        self.pipeline.best_model_checkpoint = {
            "epoch": epoch,
            "checkpoint": checkpoint
        }

    def remove(self, filename):
        del self.checkpoints[filename]


class CombinedLoss:
    """
    Loss function for representation learning. 
    Multiple losses are utilized for the respective tasks.
    In the end, the losses are linked additively.
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()
        self.npair_loss = losses.NPairsLoss(l2_reg_weight=0.02)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        
    def __call__(self, y_pred, y_true):

        embeddings, loss1_true, loss1_pred, loss2_true, loss2_pred = y_pred
        
        task1_loss = self.loss(loss1_pred, loss1_true)
        task2_loss = self.loss(loss2_pred, loss2_true)
        
        if torch.unique(y_true).flatten().size(0) > 0:
            hard_pairs = self.miner(embeddings, y_true)
            npair_loss = self.npair_loss(embeddings, y_true, hard_pairs)
            return task1_loss + task2_loss + npair_loss
        else: # very unlikely that this will ever happen
            return task1_loss + task2_loss        