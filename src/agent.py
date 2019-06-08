"""Agent"""

import os
import sys
import shutil

import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import Dataset
from net import Net
from action import Action, time_this


class Agent:
    """Agent controls training and testing model
    """
    def __init__(self):
        self.config = Config().get_config()
        self.dataset = Dataset(self.config)
        self.net = Net(self.config).get_model()
        self.action = Action(self.config)

        self.log_dir = os.path.join(self.config['log_dir'],
                                    self.config['experiment_index'])
        self.model_dir = os.path.join(self.config['model_dir'],
                                      self.config['experiment_index'])

        self.device = None
        self._set_environment()

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_acc = 0.0
        self.best_metrics = None # rely on val_metrics to be best

    def run(self):
        """Iterative trainig and evaluate model"""

        train_loader, val_loader = self.dataset.get_dataloader()
        model = self.net.to(self.device)

        learning_rate = self.config["learning_rate"]
        optimizer = self.action.get_optimizer(model, learning_rate)
        loss_fn = self.action.get_loss_fn().to(self.device)

        self._run(train_loader, val_loader, model, loss_fn, optimizer)

    def _run(self, train_loader, val_loader, model, loss_fn, optimizer):
        """Run model"""
        n_epochs = self.config["n_epochs"]
        exp = self.config["experiment_index"]
        eval_frequency = self.config["eval_frequency"]
        threshold = self.config["upper_threshold"]

        for epoch in range(1, n_epochs+1):
            print(">> Exp: {} -> Epoch {}/{}".format(exp, epoch, n_epochs))
            self.action.update_threshold(threshold, epoch)

            train_loss = self.train_epoch(train_loader, model, loss_fn,
                                          optimizer, threshold)
            print("\n>> Train Loss:{:.4f}".format(train_loss))
            sys.stdout.flush()

            if epoch % eval_frequency == 0:
                print("\n>> Exp:{} -> Eval at Epoch: {}/{}".format(
                    exp, epoch, n_epochs), flush=True)
                train_loss, train_metric = \
                    self.eval_epoch(train_loader, model, loss_fn, threshold)
                print("\n>> Eval Loss At Train:{:.4f}".format(train_loss),
                      flush=True)
                val_loss, val_metric = \
                    self.eval_epoch(val_loader, model, loss_fn, threshold)
                print("\n>> Eval Loss At Val:{:.4f}".format(val_loss),
                      flush=True)

                if self.best_acc < val_metric["acc"]:
                    self.best_acc = val_metric["acc"]
                    self.best_metrics = (train_metric, val_metric)
                    self.action.save_model(model, optimizer, epoch,
                                           self.best_metrics, last=False)

        last_metrics = (train_metric, val_metric)
        self.action.save_model(model, optimizer, n_epochs, last_metric,
                               last=True)

    @time_this
    def _set_environment(self):
        """set environment"""
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['cuda']
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        # set random seed
        seed = self.config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # remove and create logdir
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        # remove and create modeldir
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)

    @time_this
    def train_epoch(self, loader, model, loss_fn, optimizer, threshold):
        """Train by epoch
        """
        model.train()

        losses = []
        for _, (data, target) in enumerate(tqdm(loader, ncols=70, desc="train")):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            # (batch_size, num_clusters)
            features = model(data)

            # (batch_size, batch_size)
            dist_matrix = self.action.get_cos_similarity_distance(features)
            # (batch_size, batch_size), as labels
            sim_matrix = self.action.get_cos_similarity_by_threshold(dist_matrix,
                                                                     threshold)
            loss = loss_fn(dist_matrix, sim_matrix)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        average_loss = np.mean(losses)
        return average_loss

    def eval_epoch(self, loader, model, loss_fn, threshold):
        """Evaluate by epoch"""
        with torch.no_grad():
            losses = []

            model.eval()
            predicted = []
            targets = []
            for _, (data, target) in enumerate(tqdm(loader, ncols=70,
                                               desc="eval")):
                data, target = data.to(self.device), target.to(self.device)
                # (batch_size, num_clusters)
                features = model(data)

                # (batch_size, batch_size)
                dist_matrix = self.action.get_cos_similarity_distance(features)
                # (batch_size, batch_size), as labels
                sim_matrix = \
                    self.action.get_cos_similarity_by_threshold(dist_matrix,
                                                                threshold)
                loss = loss_fn(dist_matrix, sim_matrix)

                pred = torch.argmax(features, dim=1).cpu().numpy()
                predicted.extend(pred)
                losses.append(loss.item())
                targets.extend(target.cpu().numpy())

        sys.stdout.flush()
        average_loss = np.mean(losses)
        metrics = self.action.get_metrics(targets, predicted)
        return average_loss, metrics
