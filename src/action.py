"""Action"""

import sys
import os
import time
import gc
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.metrics
import numpy as np
from numba import jit

sys.path.append("../ref")
from linear_assignment_ import linear_assignment


class Action:
    """Action
    """
    def __init__(self, config: dict):
        super(Action, self).__init__()
        self.config = config
        self.optimizer = None

    def plot_loss(self, loss, epoch, writer, is_train: bool):
        """Plot loss"""
        prefixed = "Train/" if is_train else "Val/"
        writer.add_scalar(prefixed + "loss", loss, epoch)

    def plot_metrics(self, metrics: dict, epoch: int, writer, is_train: bool):
        """Plot metrics"""
        tags = ["nmi", "ari", "acc"]
        prefixed = "Train/" if is_train else "Val/"
        for tag, metric in zip(tags, metrics):
            writer.add_scalar(prefixed + tag, metrics[metric], epoch)

    def get_optimizer(self, model, learning_rate: float):
        """Get optimizer
        :param model:
        :param learning_rate:
        :return: optimizer
        """
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,
                                        alpha=0.9)
        return optimizer

    def get_loss_fn(self):
        """Get loss function, Cross entropy
        :return: loss function, mse
        """
        loss_fn = nn.MSELoss(reduction="mean")
        return loss_fn

    def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix, threshold):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > threshold, similar,
                                 dissimilar)
        return sim_matrix

    def get_generated_targets(self, model, data, threshold):
        """Get generated labels by threshold
        :param model: model
        :param data: (batch_size, channels, height, width)
        :return: data: (batch_size, num_clusters)
        """
        with torch.no_grad():
            model.eval()
            # (batch_size, num_clusters)
            features = model(data)
            # (batch_size, batch_size)
            dist_matrix = self.get_cos_similarity_distance(features)
            # (batch_size, batch_size)
            sim_matrix = self.get_cos_similarity_by_threshold(dist_matrix,
                                                              threshold)
            return sim_matrix

    def update_threshold(self, threshold: float, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.config["n_epochs"]
        if epoch != 1:
            new_threshold = threshold - (0.9 - 0.5) / n_epochs
        else:
            new_threshold = threshold
        print(">>> new threshold is {}".format(new_threshold), flush=True)
        return new_threshold

    def save_model(self, model, optimizer, epoch, metrics, last=False):
        """save model"""
        val_metric = metrics
        acc = val_metric["acc"]
        checkpoint = dict()
        checkpoint['config'] = self.config
        checkpoint['epoch'] = epoch
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint["val_metric"] = val_metric

        model_name = "Exp{}-Epoch{}-Acc{:>5.2f}".format(
            self.config['experiment_index'], epoch, acc*100.0)

        model_dir = os.path.join(self.config['model_dir'],
                                 self.config['experiment_index'])

        if not last:
            model_name = model_name + "-Best"
            for f in os.listdir(model_dir):
                if f.split('-')[-1] == 'Best':
                    best_model_before = os.path.join(model_dir, f)
                    os.remove(os.path.join(best_model_before))
                    print('>> Delete best model before: {}'.\
                            format(best_model_before), flush=True)
        else:
            model_name = model_name + "-Last"
        model_path = os.path.join(model_dir, model_name)
        torch.save(checkpoint, model_path)
        print('>> Save best model: {}'.format(model_name), flush=True)

    def get_metrics(self, y_true: list, y_pred: list):
        """Get metrics"""
        metrics = dict()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert y_true.shape == y_pred.shape
        metrics["nmi"] = get_nmi(y_true, y_pred)
        metrics["ari"] = get_ari(y_true, y_pred)
        metrics["acc"], ind = get_acc(y_true, y_pred)
        print(">>> NMI:{:.4f}\tARI:{:.4f}\tACC:{:.4f}".format(metrics["nmi"],
                                                              metrics["ari"],
                                                              metrics["acc"]),
                                                              flush=True)
        return metrics

def time_this(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        sys.stdout.flush()
        elapse = time.time() - start_time
        print(">> Function: {} costs {:.4f}s".format(func.__name__, elapse),
              flush=True)
        sys.stdout.flush()
        return ret
    return wrapper

def gcollect(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        gc.collect()
        print(">> Function: {} has been garbage collected".\
            format(func.__name__), flush=True)
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

def get_nmi(y_true, y_pred):
    """Get normalized_mutual_info_score
    """
    nmi = sklearn.metrics.normalized_mutual_info_score(\
        y_true, y_pred, average_method="arithmetic")
    return nmi

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

class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
