import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        
        super(BiGRU, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        out, _ = self.gru(x,h0)
        out = out[:, -1, :]
        
        return out
    

class ConvGRU(nn.Module):

    def __init__(self, in_channels=1, out_channels=3):
        
        super(ConvGRU, self).__init__()
        
        self.name = "ConvGRU"
        
        self.kernel_size = 4
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=4, padding=0, stride=1, bias=False)
        self.bigru = BiGRU(16, 64, 2)
        self.dense = nn.Linear(128,out_channels,bias=False)

        
    def forward(self, x):
        
        if (self.kernel_size-1)%2 == 0:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2)) 
        else: 
            x = F.pad(x, (self.kernel_size // 2 -1, self.kernel_size // 2))
        
        
        conv = self.conv(x).permute(0,2,1)
        gru = self.bigru(conv)
        dense = self.dense(gru)
        
        return torch.tanh(dense)

    def is_autoencoder(self):
        
        return False