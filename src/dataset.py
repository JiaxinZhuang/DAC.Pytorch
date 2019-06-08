"""Datasets"""
import sys

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


class Dataset:
    """Datasets to organize data and load"""
    def __init__(self, config):
        self.config = config

        self.dataset = self.config["dataset"]
        self.img_channel = None
        self.img_row = None
        self.img_col = None
        self.train_dataset, self.valid_dataset, self.targets_uniq = \
            self.load_data()
        self.train_loader, self.valid_loader = self.load_dataloader()

    def load_data(self):
        """Load specific dataset"""
        if self.dataset == "mnist":
            self.img_channel = 1
            self.img_row, self.img_col = 28, 28
            train_dataset, valid_dataset, targets_uniq = self._load_mnist()
        else:
            print(">> Error: No datasets available")
            sys.exit(-1)
        return train_dataset, valid_dataset, targets_uniq

    def _load_mnist(self):
        """Load mnist"""
        mean, std = 0.1307, 0.3081

        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10,
                                    translate=(0.1, 0.1),
                                    scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))])

        train_dataset = torchvision.datasets.MNIST(
            '../data',
            train=True,
            download=False,
            transform=train_transform)
        valid_dataset = torchvision.datasets.MNIST(
            '../data',
            train=False,
            download=False,
            transform=valid_transform)

        targets_uniq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return train_dataset, valid_dataset, targets_uniq

    def get_dataloader(self):
        """return dataloader for agent
        """
        return self.train_loader, self.valid_loader

    def load_dataloader(self):
        """Load dataloader
        """
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        return train_loader, valid_loader


        #mean = 0.13092537
        #std = 0.30844885
        #train_transform = transforms.Compose([
        #    transforms.RandomAffine(degrees=10,
        #                            translate=(0.1, 0.1),
        #                            scale=(0.95, 1.05)),
        #    transforms.ToTensor(),
        #    transforms.Normalize((mean,), (std,))
        #    ])
        #val_transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((mean,), (std,))
        #    ])

        #dataset_part1 = torchvision.datasets.MNIST('../data', train=True,
        #                                           download=True)
        #dataet_part2 = torchvision.datasets.MNIST('../data', train=False,
        #                                           download=True)
        #train_data1, train_targets1 = dataset_part1.data, dataset_part1.target
        #train_data2, train_targets2 = dataset_part2.data, dataset_part2.target
        #train_data = np.vstack((train_data1, train_data2))
        #train_target =

        #val_dataset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        #train_dataset =






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
