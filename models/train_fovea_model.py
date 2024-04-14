import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import logging

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_file, data_dir, batch_size):
    """Load data using a custom dataset generator with CSV filtering."""
    # Load the CSV file to filter images
    df = pd.read_csv(csv_file)
    valid_images = set(df['scan_name'].values)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class CustomDataset(Dataset):
        """Custom Dataset class for loading images."""
        def __init__(self, data_dir, valid_images, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if f.endswith('.jpg') and f in valid_images]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = int(img_path.split(os.sep)[-1].split('_')[0])  # Assuming filename convention
            return image, label

    dataset = CustomDataset(data_dir, valid_images, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
    data_dir = './data/train'
    epochs = 10
    batch_size = 32
    lr = 0.001
    num_classes = 2
    
    if not os.path.exists(data_dir):
        logging.error("Data directory does not exist.")
        sys.exit(1)

    train_loader = load_data(csv_file, data_dir, batch_size)
    model, device = setup_model(num_classes)
    trained_model = train(model, device, train_loader, epochs, lr)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'resnet18_model.pth')

if __name__ == "__main__":
    main()
