import sys
import os
import argparse
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from scipy.stats import moment
from time import strftime
from glob import glob
from os.path import exists, join 
from torchsummary import summary

from ood.datasets import ImageNet
from ood.datasets import Cifar10
from ood.batch_logger import BatchLogger
from ood.log import logger
from ood import log
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
        
        self.runs = sorted(glob(join(self.runs_dir,'*/')))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.continue_run == 'last' and len(self.runs) != 0:
            self.run_dir = join(self.runs_dir,self.runs[-1])

        if self.test_run =='last' and len(self.runs) != 0:
            self.run_dir=join(self.runs_dir,self.runs[-1])



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
            return torch.load(join(self.run_dir,'val_acc.pth'))
        else:
            return torch.load(join(self.run_dir,'best_val_acc.pth'))


    def _get_inital_epoch(self):
        if self.continue_run:
            with open(self.log_file, 'r') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader)  
            return row_count-1  # 1st row is header 
        else:
            return 0

    def _setup_run(self):
        if self.continue_run and not self.run_dir:
            assert exists(self.continue_run), f'{self.continue_run} does not exist'
        else:
            self.run_dir = join(self.runs_dir,self.timestamp + str(self.command))
            os.makedirs(self.run_dir)

        with open(join(self.run_dir,'command.txt'),'a+') as command:
            command.write(repr(self) + '\n')
            command.write(self.log_dir+ self.timestamp + repr( self))
        

        self.log_file= join(self.run_dir,'training_log.txt')
        log.add_output(self.log_file)

        self.initial_epoch = self._get_inital_epoch()
        self.writer = SummaryWriter(log_dir=join(self.log_dir, self.run_dir.split(os.sep)[-1]))

        with open(join(self.run_dir,'training_log.csv'), 'a+',  newline='') as file:
            writer=csv.writer(file,delimiter=',')
            writer.writerow(('Epoch','Validation Loss', 'Validation Accuracy'))

        
    def set_model_path(self, best=False):
        if best:
            self.model_path=join(self.run_dir, 'best_val_acc.pth')
        else:
            self.model_path=join(self.run_dir, 'val_acc.pth')


    def _setup_test(self):
        self.set_model_path(best=True)
        self.log_file = join(self.run_dir,'test_log.txt')
        log.add_output(self.log_file)


        with open(join(self.run_dir,'test_log.csv'), 'a+',  newline='') as file:
            writer=csv.writer(file,delimiter=',')
            writer.writerow(('Loss', 'Accuracy'))

        

    def _write_epoch_results(self, data, mode='training', epoch=None):
        mode_types=['training', 'training_ae','test']
        assert mode in mode_types, f'{mode} not supported'

        with open(join(self.run_dir, mode+'_log.csv'), 'a+',  newline='') as file:
            writer=csv.writer(file, delimiter=',')
            if mode == 'training':
                writer.writerow((epoch,)+data)
            else:
                writer.writerow(data)
    
    

    def _calc_statistics(self, hook_dict, current_batch_size):
        statistics = {}
        num_features = 3
        stats= np.empty((num_features, current_batch_size, 1))
        for layer_id, layer_hook in hook_dict.items():
            # [N,C,H,W]
            hook_copy = torch.reshape(layer_hook.features,
                                        (layer_hook.features.shape[0],
                                        layer_hook.features.shape[1],
                                        layer_hook.features.shape[2]*layer_hook.features.shape[3]))
            statistics = []
            hook_copy = hook_copy.detach()
            for i in range(3, num_features+3): #first and second standardized moment are 0 and 1, respectively
                central_moment = moment(hook_copy.cpu(), axis=2, moment=i)
                standard_dev = np.power(np.power(central_moment,2),1./i)
                standard_moment = central_moment/ standard_dev
                statistics.append(standard_moment)
            statistics = np.asarray(statistics)
            stats = np.append(stats, statistics, axis=2)
        stats = np.delete(stats, obj=[0,0] , axis=2)
        return stats




    @property
    def vgg_kwargs(self):
        return { 'num_classes': 10 }


    def _train(self, data, model=None):
        self._setup_run()
        
        model.to(self.device)

        path_val_accuracy_best_model = join(self.run_dir,'best_val_acc.pth')
        path_val_accuracy_model = join(self.run_dir,'val_acc.pth')
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                mode='max',
        #                                                patience=5,
        #                                                verbose=True
        #                                                )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


        logger.info(summary(model,(3,32,32)))
        #epoch loop:
        val_accuracy_best = 0.0
        model.train()
        for epoch in range(self.initial_epoch, self.initial_epoch+self.epochs):
            batch_logger = BatchLogger(self.initial_epoch+self.epochs, np.ceil(data.train_set_size/self.batch_size), self.writer )
            #batch loop
            running_loss_train = 0.0
            total_train =0
            correct_train = 0
            for i, train_data in enumerate(data.train_loader, 0):
                inputs, labels = train_data[0].to(self.device), train_data[1].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted==labels).sum().item()
                train_accuracy = 100 * correct_train / total_train
                batch_logger.log(epoch+1, i+1, loss.item(), train_accuracy)

            model.eval()

            #running_loss_val = 0.0
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for val_data in data.val_loader:
                    running_loss_val=0.0
                    inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                    outputs=model(inputs)
                    val_loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    

            val_accuracy_current = 100*correct_val/total_val
            scheduler.step()
            logger.info(f'Validation Accuracy: {val_accuracy_current} | Validation Loss: {val_loss}')
            
            if val_accuracy_current > val_accuracy_best:
                val_accuracy_best =val_accuracy_current
                torch.save(model.state_dict(), path_val_accuracy_best_model)
            torch.save(model.state_dict(), path_val_accuracy_model)
            epoch_results = (val_loss.cpu().numpy(),val_accuracy_current)
            self._write_epoch_results(epoch_results, mode='training', epoch=epoch)
            batch_logger.log_epoch(epoch+1, i, epoch_results[0], epoch_results[1])


    def _train_2nd(self, model_1=None, model_2=None, data=None):
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
        
        path_val_accuracy_best_model = join(self.run_dir,'best_val_acc.pth')
        path_val_accuracy_model = join(self.run_dir,'val_acc.pth')
        
        if self.loss == 'L1':
            criterion =nn.L1Loss(reduction='sum')
        elif self.loss == 'MSE':
            criterion = nn.MSELoss(reduction='sum')
        elif self.loss == 'VAE':
            criterion = loss_function   #
        optimizer = torch.optim.SGD(
            model_2.parameters(), lr=self.learning_rate, momentum=1e-4)


        batch_logger = BatchLogger(self.initial_epoch+self.epochs, np.ceil(data.train_set_size/self.batch_size), self.writer )

        #register hooks
        feature_modules = list(model_1.children())[0]
        activations={}
        for name, module in feature_modules.named_children():
            if str(module)[:5] == 'Batch':
                activations[name] = SaveFeatures(module, self.device)
        
        #foward pass through first network
        logger.info('Training second model')
        running_loss_best = 1e10
        for epoch in range(self.initial_epoch, self.initial_epoch+self.epochs):
            for batch, train_data in enumerate(data.train_loader,0):
                inputs, labels = train_data[0].to(self.device), train_data[1].to(self.device)
                outputs = model_1(inputs.double())
                ae_inputs = self._calc_statistics(activations, current_batch_size=outputs.shape[0])
                # [num_feature x N x channel]
                inner_epochs = 1
                #split ae_inputs to a- 1 batch size
                #ae_inputs = torch.reshape(ae_inputs, ( ae_inputs.shape[1], ae_inputs.shape[0],-1))
                # [N x num_features x channel]
                #intermediate_data_set = torch.utils.data.TensorDataset(ae_inputs)

                model_2.train()
                for epoch_inner in range(inner_epochs):
                    ae_inputs = torch.tensor(ae_inputs).to(self.device)
                    ae_inputs.double()
                    output = model_2(ae_inputs)
                    if self.loss == 'VAE':
                        loss = loss_function(output[0], ae_inputs, output[1], output[2])
                    else:
                        loss = criterion(output, ae_inputs.view(-1,4224))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_logger.log(epoch+1, batch+1, loss.item(), None, write_to_tensorboard=False)
                del ae_inputs


            model_2.eval()
            running_val_loss = 0
            running_val_shifted_loss = 0
            inner_epochs = 1
            with torch.no_grad():
                logger.info('Validating on validation set')
                for batch, val_data in enumerate(data.val_loader):
                    inputs, labels = val_data[0].to(self.device), val_data[1].to(self.device)
                    outputs = model_1(inputs.double())
                    ae_val_inputs = self._calc_statistics(activations, current_batch_size=outputs.shape[0])
                    ae_val_inputs = torch.tensor(ae_val_inputs).to(self.device).double()
                    output = model_2(ae_val_inputs)
                    if self.loss == 'VAE':
                        val_loss = loss_function(output[0], ae_val_inputs, output[1], output[2])
                    else:
                        val_loss = criterion(output, ae_val_inputs.view(-1,4224))
                    running_val_loss += val_loss
                    del ae_val_inputs
                
                logger.info('Validating 2nd model on validation set + uniform noise')
                for batch, val_shifted_data in enumerate(data.shifted_val_loader):
                    inputs, labels = val_shifted_data[0].to(self.device), val_shifted_data[1].to(self.device)
                    outputs = model_1(inputs.double())
                    ae_val_inputs = self._calc_statistics(activations, current_batch_size=outputs.shape[0])
                    ae_val_inputs = torch.tensor(ae_val_inputs).to(self.device).double()
                    output = model_2(ae_val_inputs)
                    if self.loss == 'VAE':
                        val_shifted_loss = loss_function(output[0], ae_val_inputs, output[1], output[2])
                    else:
                        val_shifted_loss = criterion(output, ae_val_inputs.view(-1,4224))
                    running_val_shifted_loss += val_shifted_loss
                    del ae_val_inputs

            running_val_loss = running_val_loss.cpu().numpy()
            running_val_shifted_loss = running_val_shifted_loss.cpu().numpy()

            # save model with lowest loss
            if running_val_loss < running_loss_best:
                running_loss_best = running_val_loss
                torch.save(model_2.state_dict(), path_val_accuracy_best_model)

            torch.save(model_2.state_dict(), path_val_accuracy_model)

            epoch_results = (running_val_loss, running_val_shifted_loss)
            self._write_epoch_results(epoch_results, mode='training', epoch=epoch)
            batch_logger.log_epoch_ae(epoch + 1, batch, running_val_loss, running_val_shifted_loss)

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
        model_1.load_state_dict(torch.load(self.model_path, map_location=self.device))

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(input_dim=4224)
        data = Cifar10(self.batch_size, data_dir=self.data_dir)
        self._train_2nd(model_1=model_1, model_2=model_2, data=data)

    def train_ae_imagenet(self):
        model_1 = vgg.get_model(**self.vgg_kwargs)
        model_1.load_state_dict(torch.load(self.imagenet_weights, map_location=self.device))

        if self.model_2 == 'Variational':
            model_2 = VAE()
        elif self.model_2 == 'Vanilla':
            model_2 = AutoEncoder(input_dim=4224)
        data = ImageNet(self.batch_size)
        self._train_2nd(model_1=model_1, model_2=model_2, data=data)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='command(s) to run')

    parser.add_argument('--runs_dir', default='runs', help='Directory to save runs to')
    parser.add_argument('--data-dir', default='D:/Philipp/CIFAR/par/par/datasets', help='Directory of the data')
    parser.add_argument('--log-dir', default='logs', help='Log directory for tensorboard')


    parser.add_argument('-e','--epochs', default=1, type =int, help='Epochs to run')
    parser.add_argument('-b','--batch-size',default=64, type=int, help='Batchsize')
    parser.add_argument('-lr','--learning-rate',default=0.01, type=float, help='Learning rate')
    
    parser.add_argument('--model-path',default='model_pool/vgg16_bn_cifar10.pt', help=f'Path to model weights')
    parser.add_argument('--imagenet-weights', default='D:/Philipp/data/imagenet/imagenet', help=f'Path to ImageNet directory')
    parser.add_argument('--test-run', default='last', help='which run to test. '
                      f'If none provided, uses the most recent')
    parser.add_argument('--continue-run',  nargs='?', default='', const='last',
                      help=f'continue a previous training run, given by the timestamp. '
                      f'If no run provided, continues the most recent.')


    parser.add_argument('--model-2', default='Variational', choices=['Vanilla', 'Variational'])
    parser.add_argument('-l','--loss', default='L1', choices=['L1','MSE','VAE'], help='Loss function')
    args = parser.parse_args()
    Main(args)()



class SaveFeatures():
    def __init__(self, module, device):
        self.device=device
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().to(self.device).requires_grad_(True)
    def close(self):
        self.hook.remove()