import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class OCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = "../../../../..//run/user/1002/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=run/user/1002/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share="
        img_path += self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        image = np.array(image)
        image = np.stack([image] * 3, axis=-1)  # Replicate grayscale across three channels
        label = self.dataframe.iloc[idx]['sequence_label']
        if self.transform:
            image = self.transform(image)
        return image, label

def create_balanced_loader(data, batch_size):
    # Creating balanced batches by resampling each batch to include equal number of each patient's scans
    grouped = data.groupby('patient_id')
    balanced_data = pd.DataFrame()
    samples_per_group = max(int(np.ceil(len(data) / batch_size)), 1)
    for _, group in grouped:
        resampled_group = resample(group, replace=True, n_samples=samples_per_group, random_state=42)
        balanced_data = pd.concat([balanced_data, resampled_group])
    balanced_dataset = OCTDataset(balanced_data, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]))
    loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    return loader

def modify_resnet():
    model = models.resnet18(pretrained=True)
    return model

def train_model(train_loader, val_loader, model, epochs, learning_rate, model_save_path, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Ensure data is on the same device as the model
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:  # Log every 10 batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with loss of {best_loss}')
    print('Training complete')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')
    train_data = pd.read_csv('data/data_splits/train_data.csv')
    val_data = pd.read_csv('data/data_splits/val_data.csv')
    train_loader = create_balanced_loader(train_data, batch_size=16)
    val_loader = create_balanced_loader(val_data, batch_size=16)
    model = modify_resnet()
    model = model.to(device)
    model_save_path = 'models/resnet18_fovea_model.pth' # Specify the path to save the model
    train_model(train_loader, val_loader, model, 50, 0.001, model_save_path, device)

if __name__ == "__main__":
    main()

