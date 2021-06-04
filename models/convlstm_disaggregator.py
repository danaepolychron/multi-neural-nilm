import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    
    """
    The bidirectional LSTM network.

    ...

    Attributes
    ----------
    input_size : int
        number of expected features in the input x
    hidden_size : int
        number of features in the hidden state h
    num_layers : int
        number of recurrent layers e.g., setting num_layers=2 would mean stacking two GRUs together.  
    lstm : torch.nn.LSTM
        applies a 2-layer long short-term memory (LSTM) RNN to the input sequence.
   
    """
    
    def __init__(self, input_size, hidden_size, num_layers):
        
        super(BiLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        

    def forward(self, x):
        
        # initial hidden and cell state for each element in the batch
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # keeps the 1st point of the output sequence
        output, _ = self.lstm(x,(h0,c0))
        output = output[:, 0, :]
        
        return output
    

class ConvLSTM(nn.Module):
    
    """
    The convolutional LSTM network. 

    ...

    Attributes
    ----------
    name : str
        name of the network
    kernel_size : int
        kernel size of the convolutional layer
    conv : torch.nn.Conv1D
        applies a 1-dimensional convolutional layer to the input sequence
    bigru : BiLSTM
        applies a bidirectional 2-layer long short-term memory (LSTM) RNN to the input sequence
    dense : torch.nn.Linear 
        applies a linear transformation to the input data
   
    """

    def __init__(self, in_channels=1, out_channels=3):
        
        super(ConvLSTM, self).__init__()
        
        self.name = "ConvLSTM"     
        self.kernel_size = 4
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=4, padding=0, stride=1, bias=False)
        self.bilstm = BiLSTM(input_size=16, hidden_size=64, num_layers=2)
        self.dense = nn.Linear(in_features=128, out_features=out_channels, bias=False)

        
    def forward(self, x):
        
        # the input is padded before being fed to the convolutional layer to maintain it's original size
        # similar to tf.keras padding "same"
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        #the output is reshaped to (batch, features, sequence) to enter the GRU layer
        conv = self.conv(x).permute(0,2,1)
        lstm = self.bilstm(conv)
        dense = self.dense(lstm)
        
        output = torch.tanh(dense)
        
        return output
    
    def is_autoencoder(self):
        
        return False