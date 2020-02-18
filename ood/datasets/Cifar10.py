import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms





class Cifar10(object):
    def __init__(self, batch_size=64,  data_dir='D:/Philipp/CIFAR/par/par/datasets'):
        self.data_dir=data_dir
        self.batch_size=batch_size

        #image preprocessing
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set=torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                     download=False, transform=transform)
        
        #split training set into train and val set
        split_ratio = 0.9
        train_split_size = int(split_ratio*(train_set.__len__()))
        val_split_size = train_set.__len__()-train_split_size
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [train_split_size, val_split_size])
        
        
        
        
        self.train_loader=torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4 )
        self.train_set_size = self.train_set.__len__()

        self.val_loader=torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4 )
        self.val_set_size = self.val_set.__len__()



        self.test_set=torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                                     download=False, transform=transform)
        self.test_loader=torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4 )
        self.test_set_size = self.test_set.__len__()

        
        self.classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')