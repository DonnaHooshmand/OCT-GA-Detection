import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class RepeatChannelsTransform:
    """Transform to repeat the single channel image to three channels."""
    def __call__(self, img):
        return img.repeat(3, 1, 1)

class CustomDataset(Dataset):
    """Custom Dataset class for loading images with labels from CSV file."""
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.df['image_path'] = self.df.apply(lambda row: os.path.join(data_dir, row['scan_name']), axis=1)
        self.df = self.df[self.df['image_path'].apply(os.path.exists)]  # Filter out non-existing files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)
        label = 1 if row['status'] == 'True' else 0
        return image, label

def load_data(csv_file, data_dir, batch_size):
    """Load data using the custom dataset generator with CSV filtering."""
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229]),  # Normalizing the single channel
        RepeatChannelsTransform()  # Repeating the single channel across to get 3 channels
    ])

    dataset = CustomDataset(csv_file, data_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader

def evaluate_model(model, device, test_loader):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)

    print(f'Accuracy of the model on the test images: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

def main():
    data_dir = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3'
    csv_file = './data/test_dataset.csv'
    batch_size = 32
    model_path = './models/resnet18_model_V1.pth'

    test_loader = load_data(csv_file, data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)  # Not loading pretrained weights
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjusting for 2 output classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    evaluate_model(model, device, test_loader)

if __name__ == '__main__':
    main()