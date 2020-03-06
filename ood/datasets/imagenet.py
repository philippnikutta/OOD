import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from os.path import join
from torchvision import datasets




class ImageNet(object):
    def __init__(self, batch_size=64, data_dir='D:/Philipp/data/imagenet/imagenet', workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_data = join(self.data_dir,'train')
        self.val_data = join(self.data_dir,'val')
        self.workers = workers 

        #uniform_noise_transform = transforms.Compose(
        #            [UniformNoise(),
        #            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        
        val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.train_set = datasets.ImageFolder(
            self.train_data, train_transforms
            )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)


        self.val_set = datasets.ImageFolder(
            self.val_data, val_transforms
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.val_set,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True))