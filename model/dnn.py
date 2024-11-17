import torch
import torch.nn as nn

class DenseRegressor(nn.Module):
  def __init__(self, input_size=10):
    super(DenseRegressor, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(64, 32)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(32, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    x = self.sigmoid(x)
    return x