"""Action"""

import sys
import time
import gc
from functools import wraps

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import numpy as np

sys.path.append("../ref")
from linear_assignment_ import linear_assignment


class Action:
    def __init__(self, config: dict):
        super(Action, self).__init__()
        self.config = config
        self.optimizer = None

    @time_this
    def train_epoch(self, loader, model, loss_fn, optimizer, device):
        """Train by epoch"""
        losses = []

        model.train()
        for _, (data, target) in enumerate(tqdm(loader, ncols=70, desc="train")):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            predict = model(data)
            loss = loss_fn(predict, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        torch.cuda.empty_cache()
        average_loss = np.mean(losses)
        return average_loss

    def eval_epoch(self, loader, model, loss_fn, optimizer, device):
        """Evaluate by epoch"""
        with torch.no_grad():
            losses = []

            model.eval()
            predicted = []
            targets = []
            for _, (data, target) in enumerate(tqdm(loader, ncols=70, desc="eval")):
                data, target = data.to(device), target.to(device)
                predict = model(data)
                loss = loss_fn(predict, target)
                losses.append(loss.item())
                targets.extend(target.cpu().numpy())
                predicted.extend(predict.cpu().numpy())

        torch.cuda.empty_cache()
        average_loss = np.mean(losses)
        metrics = get_metrics(targets, predicted)
        return average_loss, metrics

    def plot_loss(self, tag, loss, epoch, writter):
        """Plot loss"""
        writter.add_scalar(tag, loss, epoch)

    def plot_metrics(self, tags: list, metrics: dict, epoch: int, writter):
        """Plot metrics"""
        for tag, metric in zip(tags, metrics):
            writter.add_scalar(tag, metrics[metric], epoch)

    def get_optimizer(self, model):
        """Get optimizer
        Args:
            model: model to be optimized
        """
        learning_rate = self.config["learning_rate"]
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        return optimizer

    def get_loss_fn(self):
        """Get loss function, Cross entropy"""
        loss_fn_1 = nn.CrossEntropyLoss()
        loss_fn_2 = nn.MSELoss()
        return loss_fn_1, loss_fn_2

    def get_distance(self, features, threshold):
        """Get distance in cosine similarity with threshold
        """
        device = features.device
        similar = torch.tensor(1, device=device)
        dissimilar = torch.tensor(0, device=device)
        distance = F.cosine_similarity(features, features, dim=1, eps=1e-6)
        distance = torch.where(distance > threshold, similar, dissimilar)
        return distance

def time_this(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapse = time.time() - start_time
        print(">> Function: {} costs {:.4f}s".format(func.__name__, elapse))
        sys.stdout.flush()
        return ret
    return wrapper

def gcollect(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        gc.collect()
        print(">> Function: {} has been garbage collected".format(func.__name__))
        sys.stdout.flush()
        return ret
    return wrapper

def str2bool(val):
    """convert str to bool"""
    value = None
    if val == 'True':
        value = True
    elif val == 'False':
        value = False
    else:
        raise ValueError
    return value

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Get metrics"""
    metrics = dict()
    metrics["nmi"] = get_nmi(y_true, y_pred)
    metrics["ari"] = get_ari(y_true, y_pred)
    metrics["acc"] = get_acc(y_true, y_pred)
    return metrics

def get_nmi(y_true, y_pred):
    """Get normalized_mutual_info_score
    """
    return sklearn.metrics.normalized_mutual_info_score(y_true, y_pred)

def get_ari(y_true, y_pred):
    """Get metrics.adjusted_rand_score
    """
    return sklearn.metrics.adjusted_rand_score(y_true, y_pred)

def get_acc(y_true: np.ndarray, y_pred: np.ndarray):
    """Get acc, maximum weight matching in bipartite graphs
    ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn
               /utils/linear_assignment_.py
    """
    assert y_true.size == y_pred.size
    dim = max(y_pred.max(), y_true.max()) + 1
    cost_maxtrix = np.zeros((dim, dim), dtype=np.int64)

    for x, y in zip(y_pred, y_true):
        cost_maxtrix[x, y] += 1
        ind = linear_assignment(cost_maxtrix.max() - cost_maxtrix)

    acc = sum([cost_maxtrix[x, y] for x, y in ind]) * 1.0 / y_pred.size
    return acc, ind

#def pairwise_distances(x, y=None):
#    """
#    Input: x is a Nxd matrix
#           y is an optional Mxd matirx
#    Output: dist is a NxM matrix where dist[i,j] is the square norm
#            between x[i,:] and y[j,:] if y is not given then use 'y=x'.
#    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#    """
#    x_norm = (x ** 2).sum(1).view(-1, 1)
#
#    if y is not None:
#        y_t = torch.transpose(y, 0, 1)
#        y_norm = (y ** 2).sum(1).view(1, -1)
#    else:
#        y_t = torch.transpose(x, 0, 1)
#        y_norm = x_norm.view(1, -1)
#
#    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#    # Ensure diagonal is zero if x=y
#    if y is None:
#        dist = dist - torch.diag(dist)
#
#    return torch.clamp(dist, 0.0, np.inf)
