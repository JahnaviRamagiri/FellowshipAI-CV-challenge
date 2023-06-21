import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision

classes = [f"{i}" for i in range(0,102)]

def model_summary(model):
    """Displays the Summary of the Architecture - All the methods used and the parameters used"""
    summary(model, input_size=(3, 32, 32))

def _imshow(img):
    inv_norm = transforms.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225))
    img = inv_norm(img)      # unnormalize
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation= 'bilinear')

def display(train_loader, n= 64,):

    # get some random training images
    # dataiter = iter(train_loader)
    images, labels = next(iter(train_loader))
    for i in range(0,n,int(np.sqrt(n))):
        _imshow(torchvision.utils.make_grid(images[i: i+int(np.sqrt(n))]))
        # print labels
        plt.title(' '.join('%7s' % classes[j] for j in labels[i: i+int(np.sqrt(n))]), loc= 'left')