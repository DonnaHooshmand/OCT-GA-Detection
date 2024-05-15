import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    """Custom dataset for loading all images from a directory."""
    def __init__(self, data_dir, transform=None, label=0):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label = label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label, image_path

def load_data(data_dir, batch_size):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), dataset

def predict_and_visualize(model, device, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    filename_predictions = []
    total_images = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = outputs.softmax(dim=1)[:, 1].cpu().numpy()
            filename_predictions.extend(zip(paths, labels.cpu().numpy(), probabilities))
            total_images += len(images)

    filename_predictions.sort(key=lambda x: os.path.basename(x[0]))
    filenames, actuals, prediction_probs = zip(*filename_predictions)
    base_filenames = [os.path.basename(filename) for filename in filenames]

    plt.figure(figsize=(20, 10))
    plt.plot(base_filenames, prediction_probs, marker='o', linestyle='-', color='b', label='Probability of Positive')
    plt.scatter(base_filenames, actuals, color='r', label='Actual Labels')
    plt.xlabel('Filename')
    plt.ylabel('Prediction/Actual Label')
    plt.title('Prediction Probabilities and Actual Labels')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total number of images predicted and plotted: {total_images}")

def main():
    data_dir = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/064_OD_4_GA_6x6_SSOCT_1/500_slices'
    csv_file = './data/test_dataset.csv'
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/resnet18_model_V3.pth'

    _, dataset = load_data(data_dir, batch_size)

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predict_and_visualize(model, device, dataset)

if __name__ == '__main__':
    main()
