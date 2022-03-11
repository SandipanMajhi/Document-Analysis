import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

class Auto_encoder(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(Auto_encoder,self).__init__()
        ## Encoder part
        self.fclayer1 = nn.Linear(input_size, hidden_dims[0])
        torch.nn.init.xavier_uniform_(self.fclayer1.weight)
        self.fclayer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        torch.nn.init.xavier_uniform_(self.fclayer2.weight)
        
        ## Decoder part
        self.fclayer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        torch.nn.init.xavier_uniform_(self.fclayer3.weight)
        self.fclayer4 = nn.Linear(hidden_dims[2], input_size)
        torch.nn.init.xavier_uniform_(self.fclayer4.weight)
        
        ## activations
        self.lerelu = nn.LeakyReLU(0.2)
        
        
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
    
    def encoder(self,x):
        z = self.fclayer1(x)
        # z = F.relu(z)
        z = self.lerelu(z)
        z = self.fclayer2(z)
        z = self.lerelu(z)
        return z
    
    def decoder(self,x):
        z = self.fclayer3(x)
        z = self.lerelu(z)
        z = self.fclayer4(z)
        z = self.lerelu(z)
        return z

    
class Prediction(nn.Module):
    def __init__(self, input_size, output_size):
        super(Prediction, self).__init__()
        self.fclayer = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        z = self.fclayer(x)
        z = F.softmax(z, dim = 1)
        return z