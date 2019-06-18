"""Agent"""

import os
import shutil

import torch
import numpy as np
from tensorboardX import SummaryWriter

from dataset import Dataset
from net import Net
from action import Action, time_this


class Agent:
    """Agent controls training and testing model
    """
    def __init__(self, config: dict):
        super(Agent, self).__init__()

        self.config = config
        self.log_dir = os.path.join(self.config['log_dir'],
                                    self.config['experiment_index'])
        self.model_dir = os.path.join(self.config['model_dir'],
                                      self.config['experiment_index'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        self.set_environment()

        self.dataset = Dataset(self.config)
        self.net = Net(self.config).get_model()
        self.action = Action(self.config)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_acc = 0.0
        self.best_metrics = None # rely on val_metrics to be best

    def set_environment(self):
        """Set environment, eg. del and create file
        """
        # remove and create logdir
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        # remove and create modeldir
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)

    def run(self):
        """Iterative trainig and evaluate model"""

        train_loader, val_loader = self.dataset.get_dataloader()
        model = self.net.to(self.device)

        learning_rate = self.config["learning_rate"]
        optimizer = self.action.get_optimizer(model, learning_rate)
        loss_fn = self.action.get_loss_fn().to(self.device)

        self.fit(train_loader, val_loader, model, loss_fn, optimizer)

    def fit(self, train_loader, val_loader, model, loss_fn, optimizer):
        """Run model"""
        n_epochs = self.config["n_epochs"]
        exp = self.config["experiment_index"]
        eval_frequency = self.config["eval_frequency"]
        threshold = self.config["upper_threshold"]

        val_loss, val_metric = \
            self.eval_epoch(val_loader, model, loss_fn, threshold)
        print(">>> Eval Loss At Val:{:.4f}".format(val_loss),
              flush=True)
        print("-" * 50, flush=True)
        self.plot_epoch(val_metric, val_loss, 0, self.writer, is_train=False)

        for epoch in range(1, n_epochs+1):
            print("\n>> Exp: {} -> Train at Epoch {}/{}".format(exp, epoch, \
                  n_epochs), flush=True)

            threshold = self.action.update_threshold(threshold, epoch)

            train_loss = self.train_epoch(train_loader, model,
                                          loss_fn, optimizer, threshold)
            print(">>> Train Loss:{:.4f}".format(train_loss), flush=True)

            if epoch % eval_frequency == 0:
                print(">> Exp:{} -> Eval at Epoch: {}/{}".format(
                    exp, epoch, n_epochs), flush=True)
                val_loss, val_metric = \
                    self.eval_epoch(val_loader, model, loss_fn, threshold)
                print(">>> Eval Loss At Val:{:.4f}".format(val_loss),
                      flush=True)
                self.plot_epoch(val_metric, val_loss, epoch, self.writer,
                                is_train=False)

                if self.best_acc < val_metric["acc"]:
                    self.best_acc = val_metric["acc"]
                    self.best_metrics = val_metric
                    self.action.save_model(model, optimizer, epoch,
                                           self.best_metrics, last=False)


        last_metrics = val_metric
        self.action.save_model(model, optimizer, n_epochs, last_metrics,
                               last=True)


    @time_this
    def train_epoch(self, train_loader, model, loss_fn, optimizer, threshold):
        """Train by epoch
        """
        local_nepochs = self.config["local_nepochs"]
        local_batch_size = self.config["local_batch_size"]

        losses = []
        for _, (data, _) in enumerate(train_loader):
            # Get generated labels
            data = data.to(self.device)
            # (batch_size, batch_size)
            targets = self.action.get_generated_targets(model, data,
                                                        threshold)
            local_loader = self.dataset.get_data_with_local_batch_size(
                data, local_batch_size)

            # mode.eval is used in get_generated_targets
            model.train()
            for _ in range(local_nepochs):
                for local_data, local_index in local_loader:
                    local_data = local_data.to(self.device)
                    local_target = targets[local_index, :][:, local_index]
                    optimizer.zero_grad()
                    # (batch_size, num_clusters)
                    features = model(local_data)
                    # (batch_size, batch_size)
                    dist_matrix = \
                        self.action.get_cos_similarity_distance(features)
                    loss = loss_fn(dist_matrix, local_target)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

        average_loss = np.mean(losses)
        return average_loss

    @time_this
    def eval_epoch(self, loader, model, loss_fn, threshold):
        """Evaluate by epoch"""
        with torch.no_grad():
            model.eval()

            losses = []
            predicted = []
            targets = []
            for _, (data, target) in enumerate(loader):
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
                # (batch_size)
                pred = torch.argmax(features, dim=1).cpu().numpy()
                predicted.extend(pred)
                targets.extend(target.cpu().numpy())
                losses.append(loss.item())

        average_loss = np.mean(losses)
        metrics = self.action.get_metrics(targets, predicted)
        return average_loss, metrics

    @time_this
    def plot_epoch(self, metrics, loss, epoch, writer, is_train):
        """Plot epoch
        """
        self.action.plot_loss(loss, epoch, writer, is_train)
        self.action.plot_metrics(metrics, epoch, writer, is_train)
