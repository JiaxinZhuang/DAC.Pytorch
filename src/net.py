"""net"""

import sys
#import math
import torch.nn as nn
from torchsummary import summary

class Net:
    """Net"""
    def __init__(self, config: dict):
        super(Net, self).__init__()
        self.config = config
        self.model = None
        self._init_model()

    def _init_model(self):
        """Init model"""
        dataset = self.config["dataset"]
        if dataset == "mnist":
            self.model = MNISTNetwork(self.config)
            print(">> Net: _init_model with MNISTNetwork")
        else:
            print(">> Net: _init_model ERROR Since No dataset available")
            sys.exit(-1)

    def print_model(self, input_size):
        """print model struture given input size eg. (1, 32, 32)
        """
        assert len(input_size) == 3
        summary(self.model, input_size)

    def get_model(self):
        """Get model"""
        return self.model


class MNISTNetwork(nn.Module):
    """Mnist network"""
    def __init__(self, config: dict):
        super(MNISTNetwork, self).__init__()
        self.config = config
        # Conv layers: 1, 2, 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_af = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv2_bn = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv2_af = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_af = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool1_bn = nn.BatchNorm2d(64)
        # Conv Layers: 4, 5, 6
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv4_af = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv5_af = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv6_af = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2_bn = nn.BatchNorm2d(128)
        # Conv Layers: 7
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1)
        self.conv7_bn = nn.BatchNorm2d(10)
        self.conv7_af = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgpool_bn = nn.BatchNorm2d(10)


        # Linear
        self.linear1 = nn.Linear(10, 10)
        self.linear1_bn = nn.BatchNorm1d(10)
        self.linear1_af = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
        self.linear2_bn = nn.BatchNorm1d(10)
        self.linear2_af = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward"""
        # Conv Layers 1-7, Conv-Bn-ReLU
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.conv1_af(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.conv2_af(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = self.conv3_af(out)
        out = self.maxpool1(out)
        out = self.maxpool1_bn(out)

        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = self.conv4_af(out)
        out = self.conv4_af(out)
        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = self.conv5_af(out)
        out = self.conv6(out)
        out = self.conv6_bn(out)
        out = self.conv6_af(out)
        out = self.maxpool2(out)
        out = self.maxpool2_bn(out)

        out = self.conv7(out)
        out = self.conv7_bn(out)
        out = self.conv7_af(out)
        out = self.avgpool(out)
        out = self.avgpool_bn(out)

        # Flatten
        out = out.view(out.size(0), -1)
        # Linear Layers: 1, 2
        out = self.linear1(out)
        out = self.linear1_bn(out)
        out = self.linear1_af(out)
        out = self.linear2(out)
        out = self.linear2_bn(out)
        out = self.linear2_af(out)
        # Softmax
        out = self.softmax(out)
        return out
