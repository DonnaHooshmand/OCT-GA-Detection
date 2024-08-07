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

class LSTMOutputTarget:
    def __init__(self, index, category):
        self.index = index
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape)==2:
            return model_output[self.index, self.category]
        else:
            return model_output[:, self.index, self.category]
    
# model_path = '/home/c3labstudents/Documents/GitHub/OCT-GA-Detection/GA_Detection/GADetectionExperiments/experiments/20240716/184752/best_model/best_model.pth'

model = ResNet34_LSTM(3)
# model.load_state_dict(torch.load(model_path))
model.train()

# user_id = os.getuid()
# data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"
# data_path = '../data/experiment/01/'
# test_path = data_path + 'test.csv'
# batch_size = 1

# test_loader = data_loader(data_dir,test_path,batch_size)
print("\nTest Loader complete")
count = 0

experiment_dir = create_experiment_folders("./GA_Detection/Grad_CAM")

# with torch.no_grad():
# for images, labels, folder_names in test_loader:
images = torch.zeros((1,10,1,128,128)) #(batch_size, num_sequences, channel, width, height)
labels = torch.ones(1,10)
if torch.cuda.is_available():
    images, labels = images.cuda(), labels.cuda()
    model = model.cuda()

# Forward pass
outputs = model(images)
print(outputs.size())
batch_size, seq_len, num_classes = outputs.size()

print(batch_size)
print(seq_len)
print(num_classes)
print(outputs.shape)

# Flatten the outputs and labels for evaluation
outputs = outputs.view(batch_size, seq_len, num_classes)
labels = labels.view(batch_size, seq_len)

## Grad Cam 
        
target_layers = [model.resnet.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
cam.model = model.train()
        
targets = [LSTMOutputTarget(6,2)]
## create an input tensor image for your model
## input_tenso can be a batch tensor with several images
grayscale_cam = cam(images, targets=targets)
        
print(grayscale_cam.shape)
for i in range(grayscale_cam.shape[0]):
    im = Image.fromarray((grayscale_cam[i,:,:]*255))
    im.show()
