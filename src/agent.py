"""Agent"""

import os
import shutil

import torch
import numpy as np

from config import Config
from dataset import Dataset
from net import Net
from action import Action, time_this


class Agent:
    """Agent controls training and testing model"""
    def __init__(self):
        self.config = Config().get_config()
        self.dataset = Dataset(self.config)
        self.net = Net(self.config)
        self.action = Action(self.config)

        self.log_dir = os.path.join(self.config['log_dir'],
                                    self.config['experiment_index'])
        self.model_dir = os.path.join(self.config['model_dir'],
                                      self.config['experiment_index'])

        self.device = None
        self._set_environment()

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_acc = 0.0

    def run(self):
        """Iterative trainig and evaluate model"""

        #TODO
        train_loader, val_loader = self.dataset.get_loader()
        model = self.net.get_model().to(self.device)

        optimizer = self.action.get_optimizer(model)
        loss_fn = self.action.get_loss_fn().to(self.device)

        self._run(train_loader, val_loader, model, loss_fn, optimizer)

    def _run(self, train_loader, val_loader, model, loss_fn, optimizer):
        """Run model"""
        n_epochs = self.config["n_epochs"]
        exp = self.config["experiment_index"]
        eval_frequency = self.config["eval_frequency"]

        for epoch in range(1, n_epochs+1):
            print(">> Exp: {} -> Epoch {}/{}".format(exp, epoch, n_epochs))
            train_loss, train_metrics = \
                self.action.train_epoch(train_loader, model, loss_fn, optimizer)
            #TODO
            #self.action.plot_loss(train_loss, )

            if epoch % eval_frequency:
                print(">> Exp:{} -> Eval at Epoch: {}/{}".format(
                    exp, epoch, n_epochs))
                train_loss, train_metric = \
                    self.action.eval_epoch(train_loader, model, loss_fn)
                val_loss, val_metric = \
                    self.action.eval_epoch(val_loader, model, loss_fn)

                if self.best_acc < val_metric:
                    self.best_acc = val_metric
                    #TODO
                    self.action.save_model(model, optimizer, epoch, self.best_acc, last=False)

        self.action.save_model(model, optimizer, n_epochs, self.best_acc, last=True)

    @time_this
    def _set_environment(self):
        """set environment"""
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['cuda']
        self.device = torch.device("cuda:0")

        # set random seed
        seed = self.config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # remove and create logdir
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        # remove and create modeldir
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)
