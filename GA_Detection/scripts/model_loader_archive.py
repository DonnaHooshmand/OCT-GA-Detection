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

model_path = '/home/c3labstudents/Documents/GitHub/OCT-GA-Detection/GA_Detection/GADetectionExperiments/experiments/20240716/184752/best_model/best_model.pth'

model = ResNet34_LSTM(3)
model.load_state_dict(torch.load(model_path))
model.eval()

user_id = os.getuid()
data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"
data_path = '../data/experiment/01/'
test_path = data_path + 'test.csv'
batch_size = 1

test_loader = data_loader(data_dir,test_path,batch_size)
print("\nTest Loader complete")
count = 0

experiment_dir = create_experiment_folders("./GA_Detection/Grad_CAM")

# with torch.no_grad():
for images, labels, folder_names in test_loader:
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
        model = model.cuda()

    # Forward pass
    outputs = model(images)
    print(outputs.size())
    batch_size, seq_len, num_classes = outputs.size()
    
    # Flatten the outputs and labels for evaluation
    outputs = outputs.view(batch_size, seq_len, num_classes)
    labels = labels.view(batch_size, seq_len)
    
    ## Grad Cam 
            
    target_layers = [model.resnet.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
            
    targets = [ClassifierOutputTarget(2)]
    ## create an input tensor image for your model
    ## input_tenso can be a batch tensor with several images
    grayscale_cam = cam(images, targets=targets)
            
    # visualization = show_cam_on_image(images, grayscale_cam, use_rgb=False)
            
    model.outputs = cam.outputs
            
    # im = Image.fromarray(visualization)
    cv2.imwrite(os.path.join(experiment_dir, 'test_picture_outputs', f'{folder_names+count}.jpg'), grayscale_cam[0,:,:])
    count += 1

