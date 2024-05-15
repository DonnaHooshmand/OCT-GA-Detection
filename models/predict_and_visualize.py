import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    """Custom dataset for loading images from a CSV file with image paths and labels."""
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        df = pd.read_csv(csv_file)
        df['image_path'] = df['scan_name'].apply(lambda x: os.path.join(data_dir, x))
        df = df[df['image_path'].apply(os.path.exists)]
        self.df = df
        self.labels = df['status'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = 1 if row['status'] == True else 0
        return image, label, row['image_path']

def evaluate_model(predictions, labels, prediction_probs, filenames):
    accuracy = accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    roc_auc = roc_curve(labels, prediction_probs)
    precision, recall, _ = precision_recall_curve(labels, prediction_probs)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plotting the ROC Curve
    fpr, tpr, thresholds = roc_auc
    roc_auc_score = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plotting the Precision-Recall Curve
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Plotting the Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def predict_and_visualize(model, device, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust batch size based on your system capability
    filename_predictions = []
    total_images = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = outputs.softmax(dim=1)[:, 1].cpu().numpy()  # Assuming the second column for positive class
            # Store filename, label, and probability for each image
            filename_predictions.extend(zip(paths, labels.cpu().numpy(), probabilities))
            total_images += len(images)

    # Sort the list of tuples by filenames alphabetically
    filename_predictions.sort(key=lambda x: os.path.basename(x[0]))

    # Unzip the sorted list into separate lists
    filenames, actuals, prediction_probs = zip(*filename_predictions)
    # Extract just the base filenames without paths for plotting
    base_filenames = [os.path.basename(filename) for filename in filenames]

    # Plotting
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

    model.eval()
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    all_labels = []
    all_predictions = []
    all_probs = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = outputs.softmax(dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities)
            all_filenames.extend([os.path.basename(path) for path in paths])

    evaluate_model(all_predictions, all_labels, all_probs, all_filenames)

def main():
    data_dir = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/064_OD_4_GA_6x6_SSOCT_1/500_slices'
    csv_file = './data/test_dataset.csv'
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/resnet18_model_V3.pth'
    
    dataset = CustomDataset(csv_file, data_dir, transform=Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    predict_and_visualize(model, device, dataset)

if __name__ == '__main__':
    main()
