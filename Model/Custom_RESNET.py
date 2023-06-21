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
from Packages.data_summary import model_summary, display

def model_custom_ResNET():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  model = models.resnet50(weights='IMAGENET1K_V1')
  for params in model.parameters():
      params.requires_grad = False

  for param in model.layer3.parameters():
      param.requires_grad = True
  for param in model.layer4.parameters():
      param.requires_grad = True

  # # Print the trainable param"eters
  # for name, param in model.named_parameters():
  #     print("Trainable Parameters in Model: ")
  #     if param.requires_grad == True:
  #         print(name)



  return model

