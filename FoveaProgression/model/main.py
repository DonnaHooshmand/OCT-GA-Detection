import os
import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

import logging

from dataloader import *
from CNNLSTM import *
from train import *
from evaluate import*

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    
    data_path = './FoveaProgression/data/sample_data.csv'

    # Construct the path using the user ID from the environment to ensure it's dynamic and correct
    user_id = os.getuid()
    # data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"

    data_dir = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3'

    if not os.path.exists(data_dir):
        logging.error("Data directory does not exist.")
        sys.exit(1)
    else:
        print("Directory found, loading data ...")

    # generate data loader for training, validation and test sets
    train_loader, val_loader, test_loader = data_loader(data_path)

    model = CNNLSTM(num_classes=7)
    print(model)

    # frame_height = 128
    # frame_width = 128
    # channels = 1  # Grayscale
    # example_input = torch.rand(5, 500, channels, frame_height, frame_width)
    # output = model(example_input)
    # print(output.shape)  # Expected shape: (batch_size, frames_per_video, 7)

    if torch.cuda.is_available():
        model.cuda()  # Move model to GPU if available

    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer
    num_epochs = 10 # training epochs

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    evaluate_and_save_results(model, test_loader, './FoveaProgression/results/test_predictions.csv')

    # visualize_predictions(test_loader, model)
    # Visualize predictions on the test set with filtered labels
    # visualize_filtered_predictions(test_loader, model)


if __name__ == "__main__":
    main()
    