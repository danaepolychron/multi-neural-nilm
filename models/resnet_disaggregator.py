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
    
    def __init__(self, in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1):
        
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
    

class SimpleEncoder(nn.Module):
    
    """
    An Encoder network comprised by a 1-dimensional convolution, a batch normalisation and a dropout layer.
    Does not contain a relu activation. 

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
    
    def __init__(self, in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=1):
        
        super(SimpleEncoder, self).__init__()
        
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
        
        return self.drop(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    
    """
    A residual block.

    ...

    Attributes
    ----------
    
    encoder_X : Encoder or SimpleEncoder
        applies a 1-dimensional convolutional layer to the input sequence, followed by batch normalisation and dropout
    shortcut : SimpleEncoder
        applies a 1-dimensional convolutional layer to the original input sequence, , followed by batch normalisation and dropout
   
    """
    
    def __init__(self, in_channels=1, out_channels=3):
        
        super(ResidualBlock, self).__init__()
        
        self.encoder_x =  Encoder(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding=0)
        self.encoder_y =  Encoder(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=0)
        self.encoder_z =  SimpleEncoder(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0)
        self.shortcut = SimpleEncoder(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

        
    def forward(self, x):
        
        encoder_x = self.encoder_x(x)
        encoder_y = self.encoder_y(encoder_x)
        encoder_z = self.encoder_z(encoder_y)
        
        shortcut = self.shortcut(x)
        output = encoder_z.add(shortcut)
        
        return F.relu(output)
    

class ResNet(nn.Module):
    
    """
    The residual network.

    ...

    Attributes
    ----------
    
    residualblockX : ResidualBlock
        applies a redisual block to the input sequence
    dense : torch.nn.Linear 
        applies a linear transformation to the input data
        
    """

    def __init__(self, in_channels=1, out_channels=3):
        
        super(ResNet, self).__init__() 
        
        self.name = "ResNet"
        
        self.residualblock1 = ResidualBlock(in_channels=in_channels, out_channels=64)
        self.residualblock2 = ResidualBlock(in_channels=64, out_channels=128)
        self.residualblock3 = ResidualBlock(in_channels=128, out_channels=128)
        
        self.dense = nn.Linear(in_features=128, out_features=out_channels, bias=False)

        
    def forward(self, x):
        
        res1 = self.residualblock1(x)
        res2 = self.residualblock2(res1)
        res3 = self.residualblock3(res2)
        
        #applies global average pooling to the input 
        gap = res3.mean(2)
        
        dense = self.dense(gap)
        
        return dense
    
    def is_autoencoder(self):
        
        return False



    