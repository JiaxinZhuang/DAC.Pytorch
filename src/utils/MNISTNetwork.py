"""MNISTNetwork"""

import torch.nn as nn


class MNISTNetwork(nn.Module):
    """Mnist network"""
    def __init__(self, config: dict):
        super(MNISTNetwork, self).__init__()
        self.config = config
        bn_track_running_stats = self.config["track_running_stats"]
        bn_momentum = 0.01
        # Conv layers: 1, 2, 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(64, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv1_af = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv2_af = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv3_af = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool1_bn = nn.BatchNorm2d(64, momentum=bn_momentum,
                                          track_running_stats=bn_track_running_stats)
        # Conv Layers: 4, 5, 6
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(128, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv4_af = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(128, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv5_af = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(128, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv6_af = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2_bn = nn.BatchNorm2d(128, momentum=bn_momentum,
                                          track_running_stats=bn_track_running_stats)
        # Conv Layers: 7
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1)
        self.conv7_bn = nn.BatchNorm2d(10, momentum=bn_momentum,
                                       track_running_stats=bn_track_running_stats)
        self.conv7_af = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgpool_bn = nn.BatchNorm2d(10, momentum=bn_momentum,
                                         track_running_stats=bn_track_running_stats)
        # Linear
        self.linear1 = nn.Linear(10, 10)
        self.linear1_bn = nn.BatchNorm1d(10, momentum=bn_momentum,
                                         track_running_stats=bn_track_running_stats)
        self.linear1_af = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
        self.linear2_bn = nn.BatchNorm1d(10, momentum=bn_momentum,
                                         track_running_stats=bn_track_running_stats)
        self.linear2_af = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward"""
        # Conv Layers 1-7, Conv-Bn-ReLU
        # Details between the input and output shape, using self.print_model
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
