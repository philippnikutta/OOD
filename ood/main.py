import os
import argparse



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import moment
from time import strftime
from glob import glob
from os.path import exists, join 
from torchsummary import summary




from ood.datasets import Cifar10

from ood.log import logger


from ood.models import vgg




class Main(object):
    def __init__(self, args):
        self.args = args
        self.timestamp = strftime('%Y-%m-%d-%H.%M.%S')
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.runs = sorted(glob(join(self.runs_dir,'*/')))
        if self.continue_run =='last' and len(self.runs) != 0:
            self.run_dir=join(self.runs_dir,self.runs[-1])

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



    def _setup_run(self):
        if self.continue_run:
            assert exists(self.continue_run), f'{self.continue_run} does not exist'
        else:
            self.run_dir = join(self.runs_dir,self.timestamp + str(self.command))
            os.makedirs(self.run_dir)

        with open(join(self.run_dir,'command.txt')) as command:
            command.write(self.__repr__())
        
        
        #write log 
        #write command
        #create training csv
        
    def set_model_path(self, best=False):
        if best:
            self.model_path=join(self.run_dir, 'best_val_acc.pth')
        else:
            self.model_path=join(self.run_dir, 'val_acc.pth')


    def _setup_test(self):
        self.set_model_path(best=True)

        #create test csv
        pass
        

    def _write_batch_results(self, data):
        pass
    

    def _calc_statistics(self, hook_dict):
        statistics={}
        num_features = 5
        #stats = np.empty((num_features,#channels, #batch size))
        stats= np.empty((num_features, self.batchsize, 1))
        for layer_id, layer_hook in hook_dict.items():
            # [N,C,H,W]

            initial_shape=layer_hook.features.shape
            torch.reshape(layer_hook.features,(layer_hook.features.shape[0],
                                                layer_hook.features.shape[1],
                                                layer_hook.features.shape[2]*layer_hook.features.shape[3]))
            statistics[layer_id]=[]
            for _ in range(num_features):
                statistics[layer_id].append(moment(layer_hook.features, axis=(1,2)))
            
            
            stats.append()
        return statistics




    @property
    def vgg_kwargs(self):
        return { 'num_classes': 10 }


    def _train(self, data, model=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        #path_val_accuracy_best_model = join(self.run_dir,'best_val_acc.pth')
        #path_val_accuracy_model = join(self.run_dir,'val_acc.pth')
        
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        logger.info(summary(model,(3,32,32)))
        #epoch loop:
        val_accuracy_best = 0.0
        for epoch in range(self.epochs):
            #batch loop
            running_loss = 0.0
            model.train()

            for i, train_data in enumerate(data.train_loader, 0):
                inputs, labels = train_data[0].to(device), train_data[1].to(device)

                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                #output information of current batch
                running_loss += loss.item()
                logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>8.0f}]  Loss: {:>2.4f}'.format(
                            epoch+1, self.epochs, i+1, np.ceil(data.train_set_size/self.batch_size), running_loss                                 
                                ))
                running_loss = 0.0 
            #logger.info(f'Epoch {epoch+1} completed. Evaluating Validation accuracy...')
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for val_data in data.val_loader:
                    inputs, labels = val_data[0].to(device), val_data[1].to(device)
                    outputs=model(inputs)
                    logger.info(outputs.data[0])
                    #gets predicted class
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_accuracy_current = 100*correct/total
            logger.info(f'Validation Accuracy: {val_accuracy_current}')
            if val_accuracy_current > val_accuracy_best:
                val_accuracy_best =val_accuracy_current
                torch.save(model.state_dict(), path_val_accuracy_best_model)
            torch.save(model.state_dict(), path_val_accuracy_model)





    def train_2nd(self, model=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #use forward hooks in first module to get intermediate layer outputs
        model_1 = vgg.get_model(**self.vgg_kwargs)
        data = Cifar10(self.batch_size, data_dir=self.data_dir)

        model_1.to(device)
        logger.info(summary(model_1,(3,32,32)))


        feature_modules = list(model_1.children())[0]
        activations={}
        for name, module in feature_modules.named_children():
            if str(module)[:5] == 'Batch':
                activations[name] = SaveFeatures(module,device)
        
        
        
        model_1.eval()
        
        #foward pass through first network
        #TODO set batch size to one
        for batch, train_data in enumerate(data.train_loader,0):
            inputs, labels = train_data[0].to(device), train_data[1].to(device)
            outputs = model_1(inputs)
            logger.info(outputs.shape) # [N,outputlayer]
            # [N,C,H,W]
            for key, value in activations.items():
                logger.info(value.features)

            





        activations.close()


        


    def _test(self):
        self._setup_test()

    def train(self):
        data=Cifar10(self.batch_size, data_dir=self.data_dir)
        model=vgg.get_model(**self.vgg_kwargs)
        self._train(data, model)

            

        

class SaveFeatures():
    def __init__(self, module, device):
        self.device=device
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().to(self.device).requires_grad_(True)
    def close(self):
        self.hook.remove()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='command(s) to run')

    parser.add_argument('--runs_dir', default='runs', help='Directory to save runs to')
    parser.add_argument('--data-dir', default='D:/Philipp/CIFAR/par/par/datasets', help='Directory of the data')

    parser.add_argument('-e','--epochs', default=1, type =int, help='Epochs to run')
    parser.add_argument('-b','--batch-size',default=64, type=int, help='Batchsize')
    parser.add_argument('-lr','--learning-rate',default=0.01, help='Learning rate')

    parser.add_argument('--test-run', default='last', help='which run to test. '
                      f'If none provided, uses the most recent')
    parser.add_argument('--continue-run',  nargs='?', default='', const='last',
                      help=f'continue a previous training run, given by the timestamp. '
                      f'If no run provided, continues the most recent.')

    args = parser.parse_args()
    Main(args)()