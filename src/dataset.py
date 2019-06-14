"""Datasets"""
import sys
import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import h5py
import numpy as np
import PIL

from action import time_this


class Dataset:
    """Datasets to organize data and load"""
    def __init__(self, config):
        self.config = config

        self.data_dir = self.config["data_dir"]
        self.dataset = self.config["dataset"]
        self.filepath = os.path.join(self.data_dir, "datasets.hdf5")
        self.num_samples = None
        self.img_channel = None
        self.img_row = None
        self.img_col = None
        self.train_dataset, self.valid_dataset, self.targets_uniq = \
            self.load_data()
        self.train_loader, self.valid_loader = self.load_dataloader()

    def load_data(self):
        """Load specific dataset"""
        if self.dataset == "mnist":
            train_dataset, valid_dataset, targets_uniq = self._load_mnist()
        else:
            print(">> Error: No datasets available")
            sys.exit(-1)
        return train_dataset, valid_dataset, targets_uniq

    def _load_mnist(self):
        """Load mnist"""
        mean, std = 0.13092539, 0.3084483

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))])

        valid_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))])

        key = self.dataset
        data, targets = self.read_hdf(self.filepath, key)
        train_dataset = \
            SimpleDataset(data, targets, transform=train_transforms)
        valid_dataset = \
            SimpleDataset(data, targets, transform=valid_transforms)
        targets_uniq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return train_dataset, valid_dataset, targets_uniq

    @time_this
    def read_hdf(self, filepath: str, key:str):
        """Read hdf from filepath+key
        :param filepath: str
        :param key: name of dataset
        :return: data: (samples, channels, height, width)
        :return: targets: (samples)
        """
        with h5py.File(filepath, "r") as datasets:
            dataset = datasets[key]
            data = np.array(dataset["data"])
            targets = np.array(dataset["targets"])

        self.num_samples = data.shape[0]
        self.img_channel = data.shape[3]
        self.img_row, self.img_col = data.shape[1], data.shape[2]
        print(">> Read datasets from {}".format(filepath), flush=True)
        print(">> It has {} samples with channels:{}, height:{}, width:{}".\
                format(self.num_samples, self.img_channel, self.img_row,
                       self.img_col), flush=True)
        return data, targets

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
            pin_memory=True,
            num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers)
        return train_loader, valid_loader

    def get_data_with_local_batch_size(self, data, batch_size):
        num_workers = self.config["num_workers"]
        data = data.cpu()
        transform = transforms.Compose([
            torchvision.transforms.ToPILImage(),
            transforms.RandomAffine(degrees=[-10, 10],
                                    translate=[0.1, 0.1],
                                    scale=[0.95, 1.05],
                                    resample=PIL.Image.NEAREST),
            lambda x: self.rescale_op(x),
            transforms.ToTensor(),
            ])
        small_dataset = smallDataset(data=data, transform=transform)
        loader = torch.utils.data.DataLoader(small_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
        return loader

    def rescale_op(self, img):
        """Rescale
        :param imgs: PIL image
        """
        out = img.point(lambda i: i * 0.975)
        return out

class SimpleDataset(torch.utils.data.Dataset):
    """Simple Dataset
    """
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform):
        """Init
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        super(SimpleDataset, self).__init__()

    def __getitem__(self, index):
        """Get item
        """
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        """Len
        """
        return len(self.targets)

class smallDataset(torch.utils.data.Dataset):
    """For local repeat and shuffle data
    """
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get item
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, index

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


def get_mean_std(dataset, ratio=0.01):
    """sample with ratio to calculate mean and std
    img has (sampels, channels, height, width)
    """
    batch_size = len(dataset) * ratio
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)
    train = iter(dataloader).next()
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std
