'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchsummary import summary
from collections import OrderedDict
from Packages.data_summary import model_summary, display

def model_custom_ResNET(params = True):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = models.resnet50(weights='IMAGENET1K_V1')

  if params == True:
    for params in model.parameters():
        params.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

  num_features = model.fc.in_features

  model.fc = nn.Sequential(nn.Sequential(OrderedDict([
    ('drop', nn.Dropout(p = 0.1)),
    ('fc1', nn.Linear(num_features, 1000)),
    ('bc1', nn.BatchNorm1d(1000)),
    ('relu', nn.ReLU()),
    ('drop', nn.Dropout(p = 0.1)),
    ('fc2', nn.Linear(1000, 256)),
    ('bc2', nn.BatchNorm1d(256)),
    ('relu', nn.ReLU()),
    ('fc3', nn.Linear(256, 102)),
    ('output', nn.LogSoftmax(dim = 1))
    ])))

  return model

