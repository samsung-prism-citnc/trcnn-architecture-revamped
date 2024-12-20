import torch
import os
import torch.nn as nn
from model.cnn import CNN
from model.vit import VisionTransformer
from model.dnn import DenseRegressor

class TrCNN(nn.Module):
  def __init__(self, diffusion_x, diffusion_y, n_factor, out_d):
    super(TrCNN, self).__init__()

    D_MODEL = 9
    PATCH_SIZE = (16, 16)
    N_CHANNELS = 1
    N_HEADS = 3
    N_LAYERS = 3

    self.cnn = CNN(
        diffusion_x = diffusion_x,
        diffusion_y = diffusion_y
    )

    self.vit = VisionTransformer(
        d_model = D_MODEL,
        n_classes = n_factor,
        img_size = (diffusion_x, diffusion_y),
        patch_size = PATCH_SIZE,
        n_channels = N_CHANNELS,
        n_heads = N_HEADS,
        n_layers = N_LAYERS
    )

    self.dnn = DenseRegressor(input_size = n_factor)

  def forward(self, x):
    x = self.cnn(x)
    x = self.vit(x)
    x = self.dnn(x)
    return x
  
  def save_model(self, path):
    if not os.path.exists(path):
      os.makedirs(path)

    torch.save(self.cnn.state_dict(), path + "/cnn.pth")
    torch.save(self.vit.state_dict(), path + "/vit.pth")
    torch.save(self.dnn.state_dict(), path + "/dnn.pth")
  
  def load_model(self, path):
    self.cnn.load_state_dict(torch.load(path + "/cnn.pth", weights_only=True))
    self.vit.load_state_dict(torch.load(path + "/vit.pth", weights_only=True))
    self.dnn.load_state_dict(torch.load(path + "/dnn.pth", weights_only=True))