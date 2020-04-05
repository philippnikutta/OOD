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
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        uniform_noise_transform = transforms.Compose(
                    [UniformNoise(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set=torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                     download=False, transform=transform_train)
        
        shifted_train_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                     download=False, transform=uniform_noise_transform)
        
        #split training set into train and val set
        split_ratio = 0.9
        train_split_size = int(split_ratio * (train_set.__len__()))
        val_split_size = train_set.__len__() - train_split_size
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [train_split_size, val_split_size])
        self.shifted_train_set, self.shifted_val_set = torch.utils.data.random_split(shifted_train_set, [train_split_size, val_split_size])
        
        
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4)
        self.train_set_size = self.train_set.__len__()

        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4)
        self.val_set_size = self.val_set.__len__()




        self.test_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                                     download=False, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4)
        self.test_set_size = self.test_set.__len__()

        
        self.classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




        
        self.shifted_val_loader = torch.utils.data.DataLoader(self.shifted_val_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4)


        self.shifted_test_set = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                                     download=False, transform=uniform_noise_transform)
        self.shifted_test_loader = torch.utils.data.DataLoader(self.shifted_test_set, batch_size=self.batch_size,
                                                         shuffle=True, num_workers=4)

    def __str__(self):
        return 'Cifar10'

class UniformNoise:
    """Add uniform noise to an image."""

    def __init__(self, width=0.05):
        self.width = width

    def __call__(self, x):
        return self.uniform(x, self.width)
    
    def uniform(self, image, width=0.05, contrast_level=0.3):

            image = self.grayscale_contrast(image, contrast_level)
            return self.apply_uniform(image)


    def grayscale_contrast(self, image, contrast_factor=0.3):
        """Convert to grayscale
        :param image: 
        :returns: 
        :rtype: 
        """
        image = transforms.functional.to_grayscale(image, num_output_channels=3)
        image = transforms.functional.to_tensor(image)
        image = (1 - contrast_factor) / 2.0 + image * contrast_factor
        # image = tf.image.adjust_contrast(image, contrast_factor)
        return image

    def apply_uniform(self, x, width=0.05):
        """Apply uniform noise to the tensor x.
        :param x: a tensor
        :param width: indicates width of uniform noise
        :returns: modified tensor x
        :rtype: 
        """
        nrow = x.shape[1]
        ncol = x.shape[2]
        noise = np.random.uniform(low=-width, high=width,
                                   size=(nrow, ncol))
        noise = np.reshape(noise, (1, noise.shape[0], noise.shape[1]))
        noise = np.repeat(noise, repeats=3, axis=0)
        noise = torch.tensor(noise).double()
        return x.double() + noise