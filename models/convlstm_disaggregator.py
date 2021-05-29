import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        
        super(BiLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x,(h0,c0))
        out = out[:, -1, :]
        
        return out
    

class ConvLSTM(nn.Module):

    def __init__(self, in_channels=1, out_channels=3):
        
        super(ConvLSTM, self).__init__()
        
        self.name = "ConvLSTM"
        
        self.kernel_size = 4
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=4, padding=0, stride=1, bias=False)
        self.bilstm = BiLSTM(16, 64, 2)
        self.dense = nn.Linear(128,out_channels,bias=False)

        
    def forward(self, x):
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        
        conv = self.conv(x).permute(0,2,1)
        lstm = self.bilstm(conv)
        dense = self.dense(lstm)
        
        return torch.tanh(dense)
    
    def is_autoencoder(self):
        
        return False