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
    

class SimpleEncoder(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1):
        
        super(SimpleEncoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.1)


    def forward(self, x):
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        return self.drop(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3):
        
        super(ResidualBlock, self).__init__()
        
        self.conv_x =  Encoder(in_channels, out_channels, kernel_size=8, padding=0)
        self.conv_y =  Encoder(out_channels, out_channels, kernel_size=5, padding=0)
        self.conv_z =  SimpleEncoder(out_channels, out_channels, kernel_size=3, padding=0)
        self.shortcut = SimpleEncoder(in_channels, out_channels, kernel_size=1, padding=0)

        
    def forward(self, x):
        
        conv_x = self.conv_x(x)
        conv_y = self.conv_y(conv_x)
        conv_z = self.conv_z(conv_y)
        
        shortcut = self.shortcut(x)
        output = conv_z.add(shortcut)
        
        return F.relu(output)
    

class ResNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3):
        
        super(ResNet, self).__init__() 
        
        self.name = "ResNet"
        
        self.residualblock1 = ResidualBlock(in_channels, 64)
        self.residualblock2 = ResidualBlock(64, 128)
        self.residualblock3 = ResidualBlock(128, 128)
        
        self.dense = nn.Linear(128,out_channels,bias=False)

        
    def forward(self, x):
        
        res1 = self.residualblock1(x)
        res2 = self.residualblock2(res1)
        res3 = self.residualblock3(res2)
        dense = self.dense(res3.mean(2))
        
        return dense
    
    def is_autoencoder(self):
        
        return False



    