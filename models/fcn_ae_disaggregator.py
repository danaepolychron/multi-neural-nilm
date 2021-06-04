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
    
    def __init__(self, in_channels=3, out_channels=1, kernel_size=3, padding=1, stride=1):
        
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        
        return self.drop(self.bn(F.relu(self.conv(x))))


class Decoder(nn.Module):
    
    """
    An Decoder network comprised by a 1-dimensional transpose convolution and a batch normalisation/

    ...

    Attributes
    ----------
    
    kernel_size : int
        kernel size of the convolutional layer
    conv : torch.nn.ConvTranspose1D
        applies a 1-dimensional transpose convolutional layer to the input sequence
    bn : torch.nn.BatchNorm1d 
        applies batch normalisation to the input sequence
   
    """
    
    def __init__(self, in_channels=3, out_channels=1, kernel_size=2, stride=2):
        
        super(Decoder, self).__init__()
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        
        return F.relu(self.conv(x))
    

class FCN_AE(nn.Module):
    
    """
    The fully convolutional auto-encoder network. 

    ...

    Attributes
    ----------
    name : str
        name of the network
    encoderX : Encoder
        applies a 1-dimensional convolutional layer to the input sequence, followed by batch normalisation and dropout
    poolX : torch.nn.MaxPool1d
        applies 1D max pooling over the input data
    decoder : Decoder
        applies a 1-dimensional transpose convolutional layer to the input sequence, followed by batch normalisation
    activation : torch.nn.Conv1D
        applies a 1-dimensional convolutional layer to the input sequence
   
    """

    def __init__(self, in_channels=3, out_channels=1):
        
        super(FCN_AE, self).__init__()
        
        self.name = "FCN_AE"
        
        self.encoder1 = Encoder(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder2 = Encoder(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder3 = Encoder(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder4 = Encoder(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        

        self.decoder = Decoder(in_channels=512, out_channels=32, kernel_size=8, stride=8)

        self.activation = nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=2)

        
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        dec = self.decoder(torch.cat([enc4, enc4], dim=1))
        output = self.activation(dec)
        
        return output
    
    def is_autoencoder(self):
        
        return True