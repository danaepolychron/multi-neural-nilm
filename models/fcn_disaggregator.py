import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1):
        
        super(Encoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        return self.drop(self.bn(F.relu(self.conv(x))))


class FCN(nn.Module):

    def __init__(self, in_channels=1, out_channels=3):
        
        super(FCN, self).__init__()
        
        self.name = "FCN"
        self.encoder1 = Encoder(in_channels, 128, kernel_size=8, padding=0)
        self.encoder2 = Encoder(128, 256, kernel_size=5, padding=0)
        self.encoder3 = Encoder(256, 128, kernel_size=3, padding=0)
        self.dense = nn.Linear(128,out_channels,bias=False)

        
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        dense = self.dense(enc3.mean(2))
        
        return dense
    
    
    def is_autoencoder(self):
        
        return False



























