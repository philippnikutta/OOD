import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms





class ImageNet(object):
    def __init__(self, batch_size=64,  data_dir='D:/Philipp/data/imagenet/imagenet'):
        self.data_dir=data_dir
        self.batch_size=batch_size

        #image preprocessing
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        uniform_noise_transform = transforms.Compose(
                    [UniformNoise(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.imagenet_data = torchvision.datasets.ImageNet(self.data_dir)
        self.data_loader = torch.utils.data.DataLoader(self.imagenet_data,
                                          batch_size=self.batchsize,
                                          shuffle=True,
                                          num_workers=4)




