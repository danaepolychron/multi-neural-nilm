import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    
    """
    An Encoder network comprised by a 1-dimensional convolution, a batch normalisation and a dropout layer.

    ...

    Attributes
    ----------
    
    kernel_size : int
        kernel size of the convolutional layer
    conv : torch.nn.Conv1D
        applies a 1-dimensional convolutional layer to the input sequence
    bn : torch.nn.BatchNorm1d 
        applies batch normalisation to the input sequence
    drop : torch.nn.Dropout
        randomly zeroes some of the elements of the input tensor during training
   
    """
    
    def __init__(self, in_channels=1, out_channels=3, kernel_size=3, padding=0, stride=1):
        
        super(Encoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        
        # the input is padded before being fed to the convolutional layer to maintain it's original size
        # similar to tf.keras padding "same"
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        return self.drop(self.bn(F.relu(self.conv(x))))


class FCN(nn.Module):
    
    """
    The fully convolutional network. 

    ...

    Attributes
    ----------
    name : str
        name of the network
    encoderX : Encoder
        applies a 1-dimensional convolutional layer to the input sequence, followed by batch normalisation and dropout
    dense : torch.nn.Linear 
        applies a linear transformation to the input data
   
    """

    def __init__(self, in_channels=1, out_channels=3):
        
        super(FCN, self).__init__()
        
        self.name = "FCN"
        self.encoder1 = Encoder(in_channels=in_channels, out_channels=128, kernel_size=8, padding=0)
        self.encoder2 = Encoder(in_channels=128, out_channels=256, kernel_size=5, padding=0)
        self.encoder3 = Encoder(in_channels=256, out_channels=128, kernel_size=3, padding=0)
        self.dense = nn.Linear(in_features=128, out_features=out_channels, bias=False)

        
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        #applies global average pooling to the input 
        gap = enc3.mean(2)
        
        dense = self.dense(gap)
        
        return dense
    
    
    def is_autoencoder(self):
        
        return False



























