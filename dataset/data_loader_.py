
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from config.config_ import mean, std

class CIFAR10Data(object):
    def __init__(self, train_split=0.9):
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = val_transform
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=val_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(num_train * train_split)
        train_idx, val_idx = indices[:split], indices[split:]
        self.X_train = train_dataset.train_data[train_idx]
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
        self.test_dataset = test_dataset

    def get_train_loader(self, batch_size=128):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size,
            num_workers=2, shuffle=True
        )
        return train_loader

    def get_val_loader(self, batch_size=128):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size,
            num_workers=2, shuffle=False
        )
        return val_loader

    def get_test_loader(self, batch_size=128):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size,
            num_workers=2, shuffle=False
        )
        return test_loader

class CIFAR100Data(object):
    def __init__(self, train_split=0.9):
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = val_transform
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                   transform=val_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                    transform=test_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(num_train * train_split)
        train_idx, val_idx = indices[:split], indices[split:]
        self.X_train = train_dataset.train_data[train_idx]
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
        self.test_dataset = test_dataset

    def get_train_loader(self, batch_size=128):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size,
            num_workers=2, shuffle=True
        )
        return train_loader

    def get_val_loader(self, batch_size=128):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size,
            num_workers=2, shuffle=False
        )
        return val_loader

    def get_test_loader(self, batch_size=128):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size,
            num_workers=2, shuffle=False
        )
        return test_loader