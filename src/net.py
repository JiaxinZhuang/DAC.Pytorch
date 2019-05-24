"""net"""

import sys
import math

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision

from action import normalize


class Net:
    """Net"""
    def __init__(self, config, input_channel, output_channel, input_resolution):
        super(Net, self).__init__()
        self.config = config
        self.model_name = config['backbone']
        self.input_resolution = input_resolution
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.model = None
        self._get_model()
        self._weight_initialize()

    def _get_model(self):
        if self.model_name == 'LeNet5':
            self.model = LeNet(self.input_channel, self.output_channel)
        elif self.model_name == 'AlexNet':
            self.model = AlexNet(self.input_channel, self.output_channel)
        elif self.model_name == 'ResNet18':
            self.model = ResNet18(self.config, self.input_channel,
                                  self.output_channel, self.input_resolution)
        else:
            print("Must provide model")
            sys.exit(-1)

    def get_model(self):
        """return model"""
        return self.model

    def print_model(self, input_size):
        """print model struture given input size eg. (1, 32, 32)
        """
        assert len(input_size) == 3
        summary(self.model, input_size)

    def _weight_initialize(self):
        """Initialize weight and biases"""
        initialize_method = self.config["initialize_method"]
        if initialize_method == "kaiming_unif":
            print(">> Using kaiming uniform initialization")
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class ResNet18(nn.Module):
    """ResNet 18"""
    def __init__(self, config, input_dim, output_dim, input_resolution):
        super(ResNet18, self).__init__()
        self.config = config
        model = torchvision.models.resnet18(
            pretrained=self.config['pretrained'])
        assert input_dim == 3
        assert model.inplanes == 512

        if input_resolution == 32:
            model.conv1.stride = 1
            self.convnet = nn.Sequential(
                *(list(model.children())[0:3]),
                *(list(model.children())[4:-2]),
                nn.AdaptiveAvgPool2d(1),
                )
        else:
            self.convnet = nn.Sequential(
                    *(list(model.children())[:-1])
                )

        self.linear1 = nn.Linear(model.inplanes, 128)
        self.linear1_bn = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, self.config['embedding_len'])
        # classification_layer for cross entropy
        self.classification_layer = nn.Linear(
            self.config['embedding_len'], output_dim) \
            if self.config['method'] in ["cross_entropy"] else None

    def forward(self, x):
        out = self.get_embedding(x)
        if self.classification_layer is not None:
            out = self.classification_layer(out)
        return out

    def get_embedding(self, x):
        """Return embedding of x"""
        out = self.convnet(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear1_bn(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = normalize(out)
        return out




class AlexNet(nn.Module):
    """AlexNet"""
    def __init__(self, input_channel, output_channel):
        super(AlexNet, self).__init__()
        assert input_channel == 3
        self.conv1 = nn.Conv2d(input_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 15, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_channel)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    """LeNet"""
    def __init__(self, input_channel=32, output_channel=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_channel)
        self.maxpool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool_2d(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool_2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class mnistNetwork(nn.Module):
    """Mnist network"""
    def __init__(self, config):
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
        # Conv Linear 1, 2, 3
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
