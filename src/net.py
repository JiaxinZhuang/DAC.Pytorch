"""net"""

import sys
import math
import torch
import torch.nn as nn
from torchsummary import summary

from utils.MNISTNetwork import MNISTNetwork
from utils.CIFAR10Network import CIFAR10Network

class Net:
    """Net"""
    def __init__(self, config: dict):
        super(Net, self).__init__()
        self.config = config
        self.model, self.input_shape = self.init_model()
        self.reset_parameters()

    def init_model(self):
        """Init model"""
        dataset = self.config["dataset"]
        if dataset == "mnist":
            print(">> Net: _init_model with MNISTNetwork")
            model = MNISTNetwork(self.config)
            input_shape = (1, 28, 28)
        elif dataset == "cifar10":
            print(">> Net: _init_model with CIFAR10Network")
            model = CIFAR10Network(self.config)
            input_shape = (3, 32, 32)
        else:
            print(">> Net: _init_model ERROR Since No dataset available")
            sys.exit(-1)
        return model, input_shape

    def print_model(self, input_size):
        """print model struture given input size eg. (1, 32, 32)
        """
        assert len(input_size) == 3
        summary(self.model, input_size, device="cpu")

    def get_model(self):
        """Get model"""
        self.print_model(self.input_shape)
        return self.model

    def reset_parameters(self):
        """Reset parameters of the model
        """
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal distribution
                fan_in = module.kernel_size[0] * module.kernel_size[1] * \
                    module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / fan_in))
                module.bias.data.zero_()
                print(">> Reset parameter {}".format(module), flush=True)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                print(">> Reset parameter {}".format(module), flush=True)
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                print(">> Reset parameter {}".format(module), flush=True)
            elif isinstance(module, nn.Linear):
                torch.nn.init.eye_(module.weight.data)
                module.bias.data.zero_()
                print(">> Reset parameter {}".format(module), flush=True)
