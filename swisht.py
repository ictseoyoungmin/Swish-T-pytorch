import torch
import torch.nn as nn

class SwishT(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

class SwishT_A(nn.Module):
    def __init__(self, alpha=0.1,):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sigmoid(x)*(x+2*self.alpha)-self.alpha

class SwishT_B(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha)-self.alpha

class SwishT_C(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha/self.beta)-self.alpha/self.beta
    

if __name__ == '__main__':
    act = SwishT_C()

    x = torch.linspace(-3,3,50)
    with torch.no_grad():
        print(act(x).shape)
