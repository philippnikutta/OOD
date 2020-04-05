import torch

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super().__init__()
        if encoding_dim == None:
            self.encoding_dim = 500 
        else:
            self.encoding_dim = encoding_dim
        self.input_dim = input_dim

        self.encoder=nn.Sequential(
            nn.Linear(self.input_dim,  self.encoding_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*4,  self.encoding_dim * 3),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoding_dim * 3 ,self.encoding_dim *2),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoding_dim *2, self.encoding_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Linear(self.encoding_dim, self.encoding_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2,  self.encoding_dim * 3),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoding_dim*3, self.encoding_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoding_dim * 4, self.input_dim),
            nn.ReLU(inplace=True)
        )



    def forward(self,x):
        x=self.encoder(x.view(-1,self.input_dim))
        x=self.decoder(x)
        return x