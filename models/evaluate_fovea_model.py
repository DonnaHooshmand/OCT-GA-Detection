import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize, Lambda
from torchvision.models import resnet18
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    """Dataset class for loading images."""
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the annotations CSV file.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        label = 1 if self.img_labels.iloc[idx, 1] == 'True' else 0
        if self.transform:
            image = self.transform(image)
        return image, label

def load_model(model_path, num_classes):
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def main():
    # Transformations for the image
    transform = Compose([
        Resize((224, 224)),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229]),
        Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat grayscale image across 3 channels
    ])
    
     # Construct the path using the user ID from the environment to ensure it's dynamic and correct
    user_id = os.getuid()
    data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"

    # Data loaders
    test_dataset = CustomDataset(annotations_file='data/test_labels.csv', 
                                 img_dir= data_dir, 
                                 transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the trained model
    model = load_model('models/resnet18_model.pth', num_classes=2)

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
