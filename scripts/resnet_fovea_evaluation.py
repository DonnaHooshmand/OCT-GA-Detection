import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np

class OCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        image = np.array(image)
        image = np.stack([image] * 3, axis=-1)  # Replicate grayscale across three channels
        label = self.dataframe.iloc[idx]['sequence_label']
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(file_path, transform):
    data = pd.read_csv(file_path)
    return OCTDataset(data, transform=transform)

def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    average_loss = total_loss / len(loader)
    return accuracy, average_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # Loading trained model
    model_path = './models/resnet18_fovea_model.pth'
    model = torch.load(model_path)
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    val_dataset = load_data('data/data_splits/val_data.csv', transform)
    test_dataset = load_data('data/data_splits/test_data.csv', transform)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    val_accuracy, val_loss = evaluate_model(model, val_loader, device)
    test_accuracy, test_loss = evaluate_model(model, test_loader, device)
    
    print(f'Validation Accuracy: {val_accuracy*100:.2f}%, Validation Loss: {val_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy*100:.2f}%, Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()
