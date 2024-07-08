import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize, Lambda, CenterCrop
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps, ImageFilter
import logging
from torchvision.transforms.functional import to_pil_image, to_tensor, center_crop, gaussian_blur
import cv2
import numpy as np

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomDataset(Dataset):
    """Custom Dataset class for loading images with labels from CSV file."""
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)  # Load the entire CSV
        self.df['image_path'] = self.df.apply(lambda row: os.path.join(data_dir, row['scan_name']), axis=1)
        self.df = self.df[self.df['image_path'].apply(os.path.exists)]  # Filter out non-existing files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path'])#.convert('L')  # Open as grayscale
        if self.transform:
            image = self.transform(image)
        # Convert True/False label to 1/0
        label = 1 if row['status'] == True else 0
        return image, label

def apply_clahe(image):
    image_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_np)
    image_pil = Image.fromarray(image_clahe)
    return image_pil

def high_pass_filter(input_tensor):
    array = input_tensor.numpy().squeeze(0)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_pass_image = cv2.filter2D(array, -1, kernel)
    return torch.from_numpy(high_pass_image).unsqueeze(0).float()

def sobel_operator(x):
    x = to_pil_image(x)
    x = x.filter(ImageFilter.FIND_EDGES)
    return to_tensor(x)

def denoise_image(img):
    return gaussian_blur(img, kernel_size=[5, 5])
    
def custom_center_crop(img):
    return center_crop(img, [1200, 500])

def load_data(csv_file, data_dir, batch_size, num_samples=None):
    """Load data using the updated custom dataset generator with CSV filtering."""

    transform = Compose([
        Lambda(custom_center_crop),
        # Lambda(denoise_image),
        Resize((224, 224)),
        Lambda(apply_clahe),
        ToTensor(),
        # Lambda(sobel_operator),
        # Lambda(high_pass_filter),
        Normalize(mean=[0.485], std=[0.229]),  # Normalizing the single channel
        Lambda(lambda x: x.repeat(3, 1, 1))  # Repeating the single channel across to get 3 channels
    ])

    dataset = CustomDataset(csv_file, data_dir, transform)
    if num_samples:
        # If limiting the number of samples, adjust here
        dataset = torch.utils.data.Subset(dataset, range(num_samples))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def setup_model(num_classes):
    """Setup the ResNet-18 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model, device

def train(model, device, train_loader, epochs, lr):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    logging.info("Training complete.")
    return model

def main():
    setup_logging()

    # Hardcoded configuration parameters
    csv_file = './data/train_dataset.csv'

    # Construct the path using the user ID from the environment to ensure it's dynamic and correct
    user_id = os.getuid()
    data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"


    epochs = 50
    batch_size = 32
    lr = 0.00005
    num_classes = 2
    
    if not os.path.exists(data_dir):
        logging.error("Data directory does not exist.")
        sys.exit(1)

    train_loader = load_data(csv_file, data_dir, batch_size)
    model, device = setup_model(num_classes)
    trained_model = train(model, device, train_loader, epochs, lr)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'models/resnet18_model_V4.pth')

if __name__ == "__main__":
    main()
