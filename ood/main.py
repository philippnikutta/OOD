import sys
import os
import argparse
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import h5py
import umap

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import moment
from matplotlib import pyplot as plt
from time import strftime
from glob import glob
from os.path import exists, join
from torchsummary import summary

from ood.datasets import ImageNet
from ood.datasets import Cifar10
from ood.batch_logger import BatchLogger
from ood.log import logger
from ood import log
from ood import utils
from ood.models import vgg
from ood.models.autoencoder import AutoEncoder
from ood.models.vae import VAE
from ood.losses import loss_function


class Main(object):
    def __init__(self, args):
        self.args = args
        self.timestamp = strftime('%Y-%m-%d-%H.%M.%S')

        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.runs = sorted(glob(join(self.runs_dir, '*/')))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if self.verbosity == 1:
            log.set_verbosity(2)
        else:
            log.set_verbosity(1)

        if self.continue_run == 'last' and len(self.runs) != 0:
            self.run_dir = self.runs[-1]

        if self.test_run == 'last' and len(self.runs) != 0:
            self.run_dir = self.runs[-1]

    def __str__(self):
        return json.dumps(self.args.__dict__)

    def __repr__(self):
        return 'python main.py ' + ' '.join(sys.argv[1:])

    def __call__(self):
        if (self.command[0] == '_'
            or not hasattr(self, self.command)
                or not callable(getattr(self, self.command))):
            raise RuntimeError(f"bad command: {self.command}")
        getattr(self, self.command)()

    def load_weigths(self):
        if self.continue_run:
            return torch.load(join(self.runs[-1], 'val_acc.pth'))
        else:
            return torch.load(join(self.run_dir, 'best_val_acc.pth'))

    def _get_inital_epoch(self):
        if self.continue_run:
            with open(join(self.run_dir, 'training_log.csv'), 'r') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader)
            return row_count - 1  # 1st row is header
        else:
            return 0

    def _setup_run(self):
        if self.continue_run and not self.run_dir:
            assert exists(
                self.continue_run), f'{self.continue_run} does not exist'
        else:
            if not self.continue_run:
                run_dir_name = '_'.join([self.timestamp, self.command, 'batch_size', str(self.batch_size),
                                         'learning_rate', str(
                                             self.learning_rate),
                                         'optim', str(self.optim), 'loss', str(self.loss)])
                self.run_dir = join(self.runs_dir, run_dir_name)
                os.makedirs(self.run_dir)
        with open(join(self.run_dir, 'command.txt'), 'a+') as command:
            command.write(str(self) + '\n')
            command.write(f'{self.log_dir} {self.timestamp} {self}')

        self.log_file = join(self.run_dir, 'training_log.txt')
        log.add_output(self.log_file)

        self.initial_epoch = self._get_inital_epoch()
        self.writer = SummaryWriter(log_dir=join(
            self.log_dir, self.run_dir.split(os.sep)[-1]))

        with open(join(self.run_dir, 'training_log.csv'), 'a+',  newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if not self.continue_run:
                if self.loss == 'VAE':
                    writer.writerow(
                        ('Epoch', 'Validation Loss Sum', 'Validation Loss KLD', 'Validation Loss Reco', 'Shifted Validation Loss Sum', 'Shifted Validation Loss KLD', 'Shifted Validation Loss Reco'))
                else:
                    writer.writerow(
                        ('Epoch', 'Validation Loss', 'Validation Accuracy'))

    def set_model_path(self, best=False):
        if best:
            self.model_path = join(self.run_dir, 'best_val_acc.pth')
        else:
            self.model_path = join(self.run_dir, 'val_acc.pth')

    def _setup_test(self):
        self.set_model_path(best=True)
        self.log_file = join(self.run_dir, 'test_log.txt')
        log.add_output(self.log_file)

        self.writer = SummaryWriter(log_dir=join(
            self.log_dir, self.run_dir.split(os.sep)[-1]))

        with open(join(self.run_dir, 'test_log.csv'), 'a+',  newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(('Loss', 'Accuracy'))

    def _write_epoch_results(self, data, mode='training', epoch=None):
        mode_types = ['training', 'training_ae', 'test', 'test_ae']
        assert mode in mode_types, f'{mode} not supported'

        with open(join(self.run_dir, mode+'_log.csv'), 'a+',  newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if mode == 'training':
                writer.writerow((epoch,)+data)
            elif mode == 'training_ae':
                writer.writerow((epoch,))
            else:
                writer.writerow(data)

    def _calc_statistics(self, hook_dict, current_batch_size):
        statistics = {}
        num_features = self.number_moments
        stats = np.empty((num_features, current_batch_size, 1))
        for layer_id, layer_hook in hook_dict.items():
            # [N, C, H, W]
            hook_copy = torch.reshape(layer_hook.features,
                                      (layer_hook.features.shape[0],
                                       layer_hook.features.shape[1],
                                       layer_hook.features.shape[2]*layer_hook.features.shape[3]))
            # [N, C, H x W]
            statistics = []
            hook_copy = hook_copy.detach()
            # first and second standardized moment are 0 and 1, respectively
            for i in range(3, num_features + 3):
                central_moment = moment(hook_copy.cpu(), axis=2, moment=i)
                standard_dev = np.power(np.power(central_moment, 2), 1./i)
                standard_moment = central_moment / standard_dev
                statistics.append(standard_moment)
            statistics = np.asarray(statistics)
            stats = np.append(stats, statistics, axis=2)
        stats = np.delete(stats, obj=[0, 0], axis=2)
        return stats

    @property
    def vgg_kwargs(self):
        return {'num_classes': 10}

    def _train_(self, data, model, loss_function, optimizer, scheduler=None):
        """
        assumes data to have a test_loader and a val_loader
        """
        model.to(self.device)

        path_val_accuracy_best_model = join(self.run_dir, 'best_val_acc.pth')
        path_val_accuracy_model = join(self.run_dir, 'val_acc.pth')

        model.train()
        val_accuracy_best = 0.0
        for epoch in range(self.initial_epoch, self.initial_epoch+self.epochs):
            batch_logger = BatchLogger(self.initial_epoch+self.epochs,
                                       np.ceil(data.train_set_size /
                                               self.batch_size),
                                       self.writer)
            running_loss_train = 0.0
            total_train = 0
            correct_train = 0
            for i, train_data in enumerate(data.train_loader, 0):
                inputs, labels = train_data[0].to(
                    self.device), train_data[1].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_accuracy = 100 * correct_train / total_train
                batch_logger.log(epoch+1, i+1, loss.item(), train_accuracy)

            model.eval()

            # running_loss_val = 0.0
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for val_data in data.val_loader:
                    running_loss_val = 0.0
                    inputs = val_data[0].to(self.device),
                    labels = val_data[1].to(self.device)
                    outputs = model(inputs)
                    val_loss = loss_function(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy_current = 100*correct_val/total_val
            scheduler.step()
            logger.info(
                f'Validation Accuracy: {val_accuracy_current} | Validation Loss: {val_loss}')

            if val_accuracy_current > val_accuracy_best:
                val_accuracy_best = val_accuracy_current
                torch.save(model.state_dict(), path_val_accuracy_best_model)
            torch.save(model.state_dict(), path_val_accuracy_model)
            epoch_results = (val_loss.cpu().numpy(), val_accuracy_current)
            self._write_epoch_results(
                epoch_results, mode='training', epoch=epoch)
            batch_logger.log_epoch(
                epoch+1, i, epoch_results[0], epoch_results[1])

    def _train(self, data, model=None):
        self._setup_run()

        model.to(self.device)

        path_val_accuracy_best_model = join(self.run_dir, 'best_val_acc.pth')
        path_val_accuracy_model = join(self.run_dir, 'val_acc.pth')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                mode='max',
        #                                                patience=5,
        #                                                verbose=True
        #                                                )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 250], gamma=0.1)

        logger.info(summary(model, (3, 32, 32)))
        # epoch loop:
        val_accuracy_best = 0.0
        model.train()
        for epoch in range(self.initial_epoch, self.initial_epoch+self.epochs):
            batch_logger = BatchLogger(
                self.initial_epoch+self.epochs, np.ceil(data.train_set_size/self.batch_size), self.writer)
            # batch loop
            running_loss_train = 0.0
            total_train = 0
            correct_train = 0
            for i, train_data in enumerate(data.train_loader, 0):
                inputs, labels = train_data[0].to(
                    self.device), train_data[1].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_accuracy = 100 * correct_train / total_train
                batch_logger.log(epoch+1, i+1, loss.item(), train_accuracy)

            model.eval()

            # running_loss_val = 0.0
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for val_data in data.val_loader:
                    running_loss_val = 0.0
                    inputs, labels = val_data[0].to(
                        self.device), val_data[1].to(self.device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy_current = 100*correct_val/total_val
            scheduler.step()
            logger.info(
                f'Validation Accuracy: {val_accuracy_current} | Validation Loss: {val_loss}')

            if val_accuracy_current > val_accuracy_best:
                val_accuracy_best = val_accuracy_current
                torch.save(model.state_dict(), path_val_accuracy_best_model)
            torch.save(model.state_dict(), path_val_accuracy_model)
            epoch_results = (val_loss.cpu().numpy(), val_accuracy_current)
            self._write_epoch_results(
                epoch_results, mode='training', epoch=epoch)
            batch_logger.log_epoch(
                epoch+1, i, epoch_results[0], epoch_results[1])

    def _train_2nd(self, model_1=None, model_2=None, data=None):
        if not self.dry:
            self._setup_run()

        if model_1 == None:
            model_1 = vgg.get_model(pretrained=True, **self.vgg_kwargs)
        model_1.to(self.device)
        model_1.eval()
        logger.info(summary(model_1, (3, 32, 32)))
        model_1.double()

        if model_2 == None:
            model_2 = AutoEncoder(input_dim=200)
        if self.continue_run:
            model_2.load_state_dict(self.load_weigths())
        model_2.to(self.device).double()
        model_2.train()

        path_val_accuracy_best_model = join(self.run_dir, 'best_val_acc.pth')
        path_val_accuracy_model = join(self.run_dir, 'val_acc.pth')

        if self.loss == 'L1':
            criterion = nn.L1Loss(reduction='mean')
        elif self.loss == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.loss == 'VAE':
            criterion = loss_function   #

        if self.optim == 'Adadelta':
            optimizer = torch.optim.Adadelta(
                model_2.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif self.optim == 'SGD':
            optimizer = torch.optim.SGD(
                model_2.parameters(), lr=self.learning_rate, momentum=1e-4)

        batch_logger = BatchLogger(self.initial_epoch+self.epochs,
                                   np.ceil(data.train_set_size /
                                           self.batch_size),
                                   self.writer,
                                   self.log_interval)

        # register hooks
        feature_modules = list(model_1.children())[0]
        activations = {}
        for name, module in feature_modules.named_children():
            if str(module)[:5] == 'Batch':
                activations[name] = SaveFeatures(module, self.device)

        # foward pass through first network
        logger.info('Training second model')
        running_loss_best = 1e10
        for epoch in range(self.initial_epoch, self.initial_epoch+self.epochs):
            for batch, train_data in enumerate(data.train_loader, 0):
                inputs, labels = train_data[0].to(
                    self.device), train_data[1].to(self.device)
                outputs = model_1(inputs.double())
                ae_inputs = self._calc_statistics(activations,
                                                  current_batch_size=outputs.shape[0])
                # [num_feature x N x channel]
                inner_epochs = 1
                # split ae_inputs to a- 1 batch size
                # ae_inputs = torch.reshape(ae_inputs, ( ae_inputs.shape[1], ae_inputs.shape[0],-1))
                # [N x num_features x channel]
                # intermediate_data_set = torch.utils.data.TensorDataset(ae_inputs)

                model_2.train()
                for epoch_inner in range(inner_epochs):
                    ae_inputs = torch.tensor(ae_inputs).to(self.device)
                    ae_inputs.double()
                    output = model_2(ae_inputs)
                    if self.loss == 'VAE':
                        loss, [l1_loss, kld] = loss_function(
                            output[0], ae_inputs, output[1], output[2])
                    else:
                        loss = criterion(output, ae_inputs.view(-1, 4224))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_logger.log_ae(epoch+1, batch+1,
                                        loss, kld, l1_loss, write_to_tensorboard=True)
                del ae_inputs

            model_2.eval()
            running_val_loss = 0
            running_val_kld_loss = 0
            running_val_l1_loss = 0
            running_val_shifted_loss = 0
            running_val_shifted_kld_loss = 0
            running_val_shifted_l1_loss = 0
            inner_epochs = 1
            val_counter = 0
            with torch.no_grad():
                logger.info('Validating on validation set')
                for batch, val_data in enumerate(data.val_loader):
                    inputs, labels = val_data[0].to(
                        self.device), val_data[1].to(self.device)
                    outputs = model_1(inputs.double())
                    ae_val_inputs = self._calc_statistics(
                        activations, current_batch_size=outputs.shape[0])
                    ae_val_inputs = torch.tensor(
                        ae_val_inputs).to(self.device).double()
                    output = model_2(ae_val_inputs)
                    if self.loss == 'VAE':
                        val_loss, [val_l1_loss, val_kld_loss] = loss_function(
                            output[0], ae_val_inputs, output[1], output[2])
                    else:
                        val_loss = criterion(
                            output, ae_val_inputs.view(-1, 4224))
                    running_val_loss += val_loss
                    running_val_kld_loss += val_kld_loss
                    running_val_l1_loss += val_l1_loss
                    logger.info(f'Val loss: {val_loss}')
                    val_counter += 1
                    del ae_val_inputs
                logger.info(f'Val Counter: {val_counter}')
                logger.info(f'Running Val loss: {running_val_loss}')
                logger.info(
                    'Validating 2nd model on validation set + uniform noise')
                for batch, val_shifted_data in enumerate(data.shifted_val_loader):
                    inputs = val_shifted_data[0].to(self.device)
                    labels = val_shifted_data[1].to(self.device)
                    outputs = model_1(inputs.double())
                    ae_val_inputs = self._calc_statistics(
                        activations, current_batch_size=outputs.shape[0])
                    ae_val_inputs = torch.tensor(
                        ae_val_inputs).to(self.device).double()
                    output = model_2(ae_val_inputs)
                    if self.loss == 'VAE':
                        val_shifted_loss, [val_shifted_l1_loss, val_shifted_kld_loss] = loss_function(
                            output[0], ae_val_inputs, output[1], output[2])
                    else:
                        val_shifted_loss = criterion(
                            output, ae_val_inputs.view(-1, 4224))
                    running_val_shifted_loss += val_shifted_loss
                    running_val_shifted_kld_loss += val_shifted_kld_loss
                    running_val_shifted_l1_loss += val_shifted_l1_loss
                    del ae_val_inputs

                logger.info(
                    f'Running Val Shifted loss: {running_val_shifted_loss}')
            val_kld_loss = (running_val_kld_loss.cpu().numpy()/val_counter)
            val_l1_loss = (running_val_l1_loss.cpu().numpy()/val_counter)
            val_loss = (running_val_loss.cpu().numpy()/val_counter)
            val_shifted_kld_loss = (
                running_val_shifted_kld_loss.cpu().numpy()/val_counter)
            val_shifted_l1_loss = (
                running_val_shifted_l1_loss.cpu().numpy()/val_counter)
            val_shifted_loss = (
                running_val_shifted_loss.cpu().numpy()/val_counter)

            # save model with lowest loss
            if running_val_loss < running_loss_best:
                running_loss_best = running_val_loss
                torch.save(model_2.state_dict(), path_val_accuracy_best_model)

            torch.save(model_2.state_dict(), path_val_accuracy_model)

            epoch_results = (val_loss, val_kld_loss, val_l1_loss,
                             val_shifted_loss, val_shifted_kld_loss, val_l1_loss)
            self._write_epoch_results(
                epoch_results, mode='training', epoch=epoch)
            batch_logger.log_epoch_ae(epoch + 1, val_loss, val_shifted_loss)

        for key, value in activations.items():
            value.close()
        logger.info('Hooks removed')

    def _test_2nd(self, data, model_1, model_2, shifted=False):

        running_loss = 0
        running_kld_loss = 0
        running_l1_loss = 0
        # test on 1 shift
        # register hooks
        feature_modules = list(model_1.children())[0]
        activations = {}
        for name, module in feature_modules.named_children():
            if str(module)[:5] == 'Batch':
                activations[name] = SaveFeatures(module, self.device)
        logger.info(f'Testing 2nd model')

        batch_logger = BatchLogger(1,
                                   np.ceil(data.test_set_size/self.batch_size),
                                   self.writer,
                                   self.log_interval)

        if shifted:
            loader = data.shifted_test_loader
        else:
            loader = data.test_loader

        model_1.eval()
        model_2.eval()
        test_counter = 0
        with torch.no_grad():
            for batch, test_data in enumerate(loader, 0):
                inputs = test_data[0].to(self.device)
                labels = test_data[1].to(self.device)
                outputs = model_1(inputs.double())
                ae_test_inputs = self._calc_statistics(
                    activations, current_batch_size=outputs.shape[0])
                ae_test_inputs = torch.tensor(
                    ae_test_inputs).to(self.device).double()
                output = model_2(ae_test_inputs)
                if self.loss == 'VAE':
                    loss, [l1_loss, kld_loss] = loss_function(
                        output[0], ae_test_inputs, output[1], output[2])
                else:
                    loss = loss_function(output, ae_test_inputs.view(-1, 4224))
                running_loss += loss
                running_kld_loss += kld_loss
                running_l1_loss += l1_loss
                # loss = int(loss/outputs.shape[0])
                batch_logger.log_ae(1, batch, loss, kld_loss, l1_loss,
                                    write_to_tensorboard=False)

                test_counter += 1
                del ae_test_inputs, output, outputs, inputs, loss

        mean_loss = (running_loss.cpu().numpy() / test_counter)
        mean_kld_loss = (running_kld_loss.cpu().numpy()/test_counter)
        mean_l1_loss = (running_l1_loss.cpu().numpy()/test_counter)
        self._write_epoch_results(
            data=(mean_loss, mean_kld_loss, mean_l1_loss), mode='test')

        for key, value in activations.items():
            value.close()
        logger.info('Hooks removed')

    def _test(self):
        self._setup_test()

    def test_shifted(self):
        data = Cifar10(self.batch_size, data_dir=self.data_dir)
        loader = data.shifted_val_loader
        logger.info(next(iter(loader)))

    def train(self):
        data = Cifar10(self.batch_size, data_dir=self.data_dir)
        model = vgg.get_model(**self.vgg_kwargs)
        self._train(data, model)

    def train_ae_cifar10(self):
        model_1 = vgg.get_model(**self.vgg_kwargs)
        model_1.load_state_dict(torch.load(
            self.model_weights, map_location=self.device))

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(input_dim=4224)
        data = Cifar10(self.batch_size, data_dir=self.data_dir)
        self._train_2nd(model_1=model_1, model_2=model_2, data=data)

    def train_ae_imagenet(self):
        model_1 = vgg.get_model(**self.vgg_kwargs, pretrained=True)
        if self.imagenet_weights:
            model_1.load_state_dict(torch.load(
                self.imagenet_weights, map_location=self.device))

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(
                input_dim=4224, encoding_dim=self.encoding_dim)
        data = ImageNet(self.batch_size)
        self._train_2nd(model_1=model_1, model_2=model_2, data=data)

    def test_ae(self):
        self._setup_test()

        data = Cifar10(self.batch_size, data_dir=self.data_dir)

        model_1 = vgg.get_model(**self.vgg_kwargs)
        model_1.load_state_dict(torch.load(
            self.model_weights, map_location=self.device))
        model_1.to(self.device).double()

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(input_dim=4224)

        model_2.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        model_2.to(self.device).double()
        self._test_2nd(data, model_1, model_2, shifted=False)

    def test_ae_shifted(self):
        self._setup_test()

        data = Cifar10(self.batch_size, data_dir=self.data_dir)

        model_1 = vgg.get_model(**self.vgg_kwargs)
        model_1.load_state_dict(torch.load(
            self.model_weights, map_location=self.device))
        model_1.to(self.device).double()

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(input_dim=4224)

        model_2.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        model_2.to(self.device).double()
        self._test_2nd(data, model_1, model_2, shifted=True)

    def visualize_batch(self):
        data = Cifar10(self.batch_size, data_dir=self.data_dir)
        images, labels = next(iter(data.val_loader))
        logger.info(images.shape)
        grid = torchvision.utils.make_grid(images)
        logger.info(grid.shape)
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.show()
        writer = SummaryWriter(log_dir=join(self.log_dir, self.command))
        writer.add_image(tag='Val Data + Uniform Noise', img_tensor=grid)

    def create_training_set(self):
        """
        Creates dataset by forward passing original training, val, test dataset
        through classifier.

        """
        data = Cifar10(self.batch_size, data_dir=self.data_dir)

        save_path = join('datasets', 'activations',
                         '.'.join([str(data), 'hdf5']))

        model_1 = vgg.get_model(**self.vgg_kwargs)
        model_1.load_state_dict(torch.load(
            self.model_weights, map_location=self.device))
        model_1.to(self.device).double()

        logger.info(f'Creating dataset of {data}')

        # register hooks
        feature_modules = list(model_1.children())[0]
        activations = {}
        logger.info('Setting hooks')
        for name, module in feature_modules.named_children():
            if str(module)[:5] == 'Batch':
                activations[name] = SaveFeatures(module, self.device)
        logger.info(f'Hooks set')

        dataset = np.empty((self.number_moments, self.batch_size, 4224))
        # dataset = np.expand_dims(dataset, axis=-1)

        for batch, train_data in enumerate(data.train_loader, 0):
            inputs, labels = train_data[0].to(
                self.device), train_data[1].to(self.device)
            outputs = model_1(inputs.double())
            ae_inputs = self._calc_statistics(
                activations, current_batch_size=outputs.shape[0])

            dataset = np.append(dataset, ae_inputs, axis=1)
            logger.info(f'Batch: {batch}, dataset shape: {dataset.shape}')
            del ae_inputs

        path = r'D:\Philipp\projects\generalization\ood\ood\datasets\hf5'
        utils.save_hf5(dataset, join(path, 'b_64_cifar10_id.hd5'))
        for key, value in activations.items():
            value.close()
        logger.info('Hooks removed')

    def clustering(self):
        # create input data:
        logger.debug('clustering')
        data = Cifar10(batch_size=self.batch_size)
        model = vgg.get_model(**self.vgg_kwargs)
        model.load_state_dict(torch.load(
            self.model_weights, map_location=self.device))
        model.to(self.device).double()

        # identity transform
        id_tensor_path = 'ood/datasets/cluster/id_tensor.npy'
        shifted_tensor_path = 'ood/datasets/cluster/shifted_tensor.npy'

        if not exists(id_tensor_path) or not exists(shifted_tensor_path):
            feature_modules = list(model.children())[0]
            activations = {}
            for name, module in feature_modules.named_children():
                if str(module)[:5] == 'Batch':
                    activations[name] = SaveFeatures(module, self.device)
            logger.info('Hooks set')

            id_tensor = torch.tensor(
                np.empty((self.number_moments, 1, 4224))).double().to(self.device)
            shifted_tensor = torch.tensor(
                np.empty((self.number_moments, 1, 4224))).double().to(self.device)
            with torch.no_grad():
                for batch, val_data in enumerate(data.val_loader, 0):
                    inputs = val_data[0].to(self.device)
                    labels = val_data[1].to(self.device)
                    outputs = model(inputs.double())
                    ae_inputs = self._calc_statistics(
                        activations, current_batch_size=outputs.shape[0])
                    ae_inputs = torch.tensor(
                        ae_inputs).to(self.device).double()
                    # shape : Num_Moments x batch_size x num_channels
                    id_tensor = torch.cat((id_tensor, ae_inputs), axis=1)
                    del ae_inputs
                id_tensor_np = id_tensor.cpu().numpy()
                np.save('ood/datasets/cluster/id_tensor.npy', id_tensor_np)
                del id_tensor
                logger.info(data.shifted_val_set.__len__())
                for batch, shifted_val_data in enumerate(data.shifted_val_loader, 0):
                    inputs = shifted_val_data[0].to(self.device)
                    labels = shifted_val_data[1].to(self.device)
                    outputs = model(inputs.double())
                    ae_shifted_inputs = self._calc_statistics(
                        activations, current_batch_size=outputs.shape[0])
                    ae_shifted_inputs = torch.tensor(
                        ae_shifted_inputs).to(self.device).double()
                    shifted_tensor = torch.cat(
                        (shifted_tensor, ae_shifted_inputs), axis=1)
                    del ae_shifted_inputs
                logger.info('success')
                shifted_tensor_np = shifted_tensor.cpu().numpy()
                del shifted_tensor
                for key, value in activations.items():
                    value.close()
                logger.info('Hooks removed')
                np.save('ood/datasets/cluster/shifted_tensor.npy', shifted_tensor_np)
        else:
            id_tensor_np = np.load('ood/datasets/cluster/id_tensor.npy')
            shifted_tensor_np = np.load(
                'ood/datasets/cluster/shifted_tensor.npy')
            id_tensor = torch.tensor(id_tensor_np).double().to(self.device)
            shifted_tensor = torch.tensor(
                shifted_tensor_np).double().to(self.device)

        logger.info(id_tensor_np.shape)
        logger.info(shifted_tensor_np.shape)
        if not exists('ood/datasets/cluster/input_data_small.npy'):
            input_data_small = np.concatenate(
                (id_tensor_np[:100], shifted_tensor_np[:100]), axis=1).astype(np.float32)
            input_data_small = np.transpose(input_data_small, (0, 2, 1))
            input_data_small = np.reshape(input_data_small, (-1, input_data_small.shape[-1]))
            input_data_small = np.clip(input_data_small, np.finfo(np.float32).min, np.finfo(np.float32).max)
            np.save('ood/datasets/cluster/input_data_small.npy', input_data_small)
        else:
            input_data_small = np.load('ood/datasets/cluster/input_data_small.npy')

        input_data = np.concatenate(
            (id_tensor_np, shifted_tensor_np), axis=1).astype(np.float32)
        input_data = np.transpose(input_data, (0, 2, 1))
        input_data = np.reshape(input_data, (input_data.shape[-1],-1))
        input_data = np.clip(input_data, np.finfo(np.float32).min, np.finfo(np.float32).max)
        if not exists('ood/datasets/cluster/input_data.csv'):
            logger.debug('Saving to csv...')
            np.savetxt('ood/datasets/cluster/input_data.csv', input_data, delimiter=',')

        if not exists('ood/datasets/cluster/input_data.tsv'):
            logger.debug('Saving to tsv...')
            np.savetxt('ood/datasets/cluster/input_data.tsv', input_data, delimiter='\t')

        #Target for plotting
        target = np.concatenate((np.ones((id_tensor_np.shape[1], 1)), np.zeros(
            (id_tensor_np.shape[1], 1))), axis=0)

        logger.info('Running UMAP')
        #UMAP
        standard_embedding = umap.UMAP(n_neighbors=30,
                                        min_dist=0.0,
                                        n_components=2,
                                        random_state=42,).fit_transform(input_data)

        logger.debug('saving embedding')
        np.save('ood/datasets/cluster/embedding.npy', standard_embedding)
        #No target
        logger.info('Plotting Embedding')
        plt.scatter(
            standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
        plt.show()


    def vis_dis(self):
           # create input data:
            logger.debug('Visualizing Distributions')
            data = Cifar10(batch_size=self.batch_size)
            model = vgg.get_model(**self.vgg_kwargs)
            model.load_state_dict(torch.load(
                self.model_weights, map_location=self.device))
            model.to(self.device).double()

            # identity transform
            id_tensor_path = 'ood/datasets/cluster/id_tensor.npy'
            shifted_tensor_path = 'ood/datasets/cluster/shifted_tensor.npy'

            if not exists(id_tensor_path) or not exists(shifted_tensor_path):
                feature_modules = list(model.children())[0]
                activations = {}
                for name, module in feature_modules.named_children():
                    if str(module)[:5] == 'Batch':
                        activations[name] = SaveFeatures(module, self.device)
                logger.info('Hooks set')

                id_tensor = torch.tensor(
                    np.empty((self.number_moments, 1, 4224))).double().to(self.device)
                shifted_tensor = torch.tensor(
                    np.empty((self.number_moments, 1, 4224))).double().to(self.device)
                with torch.no_grad():
                    for batch, val_data in enumerate(data.val_loader, 0):
                        inputs = val_data[0].to(self.device)
                        labels = val_data[1].to(self.device)
                        outputs = model(inputs.double())
                        ae_inputs = self._calc_statistics(
                            activations, current_batch_size=outputs.shape[0])
                        ae_inputs = torch.tensor(
                            ae_inputs).to(self.device).double()
                        # shape : Num_Moments x batch_size x num_channels
                        id_tensor = torch.cat((id_tensor, ae_inputs), axis=1)
                    logger.info(data.shifted_val_set.__len__())
                    for batch, shifted_val_data in enumerate(data.shifted_val_loader, 0):
                        inputs = shifted_val_data[0].to(self.device)
                        labels = shifted_val_data[1].to(self.device)
                        outputs = model(inputs.double())
                        ae_shifted_inputs = self._calc_statistics(
                            activations, current_batch_size=outputs.shape[0])
                        ae_shifted_inputs = torch.tensor(
                            ae_inputs).to(self.device).double()
                        shifted_tensor = torch.cat(
                            (shifted_tensor, ae_shifted_inputs), axis=1)
                    logger.info('success')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='command(s) to run')
    
    parser.add_argument('-v','--verbosity', default=0 , type=int, help='set to 1 for debug messages')
    parser.add_argument('--runs_dir', default='runs',
                        help='Directory to save runs to')
    parser.add_argument(
        '--data-dir', default='D:/Philipp/CIFAR/par/par/datasets', help='Directory of the data')
    parser.add_argument('--log-dir', default='logs',
                        help='Log directory for tensorboard')
    parser.add_argument('-e', '--epochs', default=1,
                        type=int, help='Epochs to run')
    parser.add_argument('-b', '--batch-size', default=64,
                        type=int, help='Batchsize')
    parser.add_argument('-lr', '--learning-rate',
                        default=0.01, type=float, help='Learning rate')
    parser.add_argument('--dry', action='store_true',
                        help='dry run, that does not create any folders')

    parser.add_argument('--log-interval', default=10,
                        type=int, help=f'logger ouput interval')
    parser.add_argument(
        '--model-weights', default='model_pool/vgg16_bn_cifar10.pt', help=f'Path to model weights')
    parser.add_argument('--imagenet-weights', default='D:/Philipp/data/imagenet/imagenet',
                        help=f'Path to ImageNet directory')

    parser.add_argument('--test-run', default='last', help='which run to test. '
                        f'If none provided, uses the most recent')
    parser.add_argument('--continue-run',  nargs='?', default='', const='last',
                        help=f'continue a previous training run, given by the timestamp. '
                        f'If no run provided, continues the most recent.')

    parser.add_argument('--encoding-dim', default=500, type=int,
                        help=f'dimensionality of latent representation of autoencoder')
    parser.add_argument('--number-moments', default=5,
                        type=int, help=f'how many moments to use')
    parser.add_argument('--model-2', default='Variational',
                        choices=['Vanilla', 'Variational'])
    parser.add_argument('-l', '--loss', default='VAE',
                        choices=['L1', 'MSE', 'VAE'], help='Loss function')
    parser.add_argument('-o', '--optim', default='Adadelta',
                        choices=['SGD', 'Adadelta'], help='which optimzer to use')
    args = parser.parse_args()
    Main(args)()


class SaveFeatures():
    def __init__(self, module, device):
        self.device = device
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.clone().to(self.device).requires_grad_(True)

    def close(self):
        self.hook.remove()
