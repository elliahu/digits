import numpy as np
import torch
import torchvision
import os.path
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from utils import view_classify
from model import Model
from PIL import Image, ImageDraw
from draw import Drawable

BATCH_SIZE = 64

# define the transformations that will be performed on images
# transforms.ToTensor converts images to numbers and then to Tensors
# transforms.Normalize normalizes tensors
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])


# Download data and transform it
trainset = datasets.MNIST('data_train', download=True, train=True, transform=transform)
valset = datasets.MNIST('data_val', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=True)


# Exploratory analysis
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)


# Build the neural network
# Input layer (784) -> Hidden layer (128) -> Hidden layer (64) -> Output layer (10)
INPUT_SIZE = 784
HIDDEN_SIZE = [128, 64]
OUTPUT_SIZE = 10
MODEL_PATH = './model.pt'

model = Model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE, trainloader)

""" while True:
    drawable = Drawable(28, 28, 15, 'black', 'white', brush_size=10)
    drawable.start()
    model.eval_single(drawable.create_tensor(28,28)) """

model.evaluate(valloader, BATCH_SIZE)