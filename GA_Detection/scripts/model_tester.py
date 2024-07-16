import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataloader import *
from CNNLSTM import *
from train import *
from evaluate import*
from log import *
from PIL import Image
import cv2

num_classes=3 #number of classes
# model = CNNLSTMSeq2Seq(num_classes)
model = ResNet34_LSTM(num_classes)

target_layers = [model.resnet.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(2)]

## create an input tensor image for your model
## input_tenso can be a batch tensor with several images cv2.resize(np.float32(img), target_size)
images = torch.rand(1, 1, 1, 64, 64)  # Assuming input images are 128x128
grayscale_cam = cam(images, targets=targets)
print(grayscale_cam)

# visualization = show_cam_on_image(images, grayscale_cam)

model.outputs = cam.outputs
cv2.imshow('image', grayscale_cam[0,:,:])
cv2.waitKey(0)
