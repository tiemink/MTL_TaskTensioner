import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SharedLayer(nn.Module):
    def __init__(self):
        super(SharedLayer, self).__init__()
        self.lin1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        return x

class SpecificLayer(nn.Module):
	def __init__(self):
		super(SpecificLayer, self).__init__()
		self.fc1 = nn.Linear(2,1)

	def forward(self, x):
		x = self.fc1(x)
		return x