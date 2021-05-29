import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    
    def __init__(self, in_features=3, out_features=1, kernel_size=3, padding=1, stride=1):
        
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        
        return self.drop(self.bn(F.relu(self.conv(x))))


class Decoder(nn.Module):
    
    def __init__(self, in_features=3, out_features=1, kernel_size=2, stride=2):
        
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return F.relu(self.conv(x))
    

class FCN_AE(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        
        super(FCN_DAE, self).__init__()
        
        self.name = "FCN_AE"
        
        self.encoder1 = Encoder(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder2 = Encoder(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder3 = Encoder(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder4 = Encoder(128, 256, kernel_size=3, padding=1)
        

        self.decoder = Decoder(512, 32, kernel_size=8, stride=8)

        self.activation = nn.Conv1d(32, out_channels, kernel_size=1, padding=2)

        
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        dec = self.decoder(torch.cat([enc4, enc4], dim=1))
        act = self.activation(dec)
        
        return act
    
    def is_autoencoder(self):
        
        return True