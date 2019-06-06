"""Datasets"""
import sys

import torch
from torch.utils.data import DataLoader


class Dataset:
    """Datasets to organize data and load"""
    def __init__(self, config):
        self.config = config

        self.dataset = self.config["dataset"]

        self.img_channel = None
        self.img_row = None
        self.img_col = None
        self.load_data()

    def load_data(self):
        """Load specific dataset"""
        if self.dataset == "MNIST":
            self._load_mnist()
        else:
            print(">> Error: No datasets available")
            sys.exit(-1)

    def _load_mnist(self):
        self.img_channel = 1
        self.img_row, self.img_col = 28, 28

        train_transform = transforms.Compose()






class DataPrefetcher:
    """Prefetch data each iteration, may use
    double GPU storage for iter data. And
    to tensor normalize data because
    dataloader too slow on these operations"""
    def __init__(self, loader: DataLoader, mean: list, std: list):
        """
        Args:
            loader (DataLoader): data loader
            mean (list): mean value for each channel
            std (list): std value for each channel
        """
        super(DataPrefetcher, self).__init__()
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # RGB channels
        self.mean = torch.new_tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.new_tensor(std).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        """Preload data if available, or return None"""
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        """Next method
        1. wait asynchronous call
        2. run next call without wait
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        target = self.next_target
        self.preload()
        return inputs, target
