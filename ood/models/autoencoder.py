import torch

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(),
            nn.ReLU(inplace=True)
            nn.Linear(),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Linear(),
            nn.ReLU(inplace=True),
            nn.Linear(),
            nn.ReLU(inplace=True)
        )



    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x