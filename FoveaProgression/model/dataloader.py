import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class EyeScanDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset.
        :param images: List of numpy arrays, each representing a sequence of images.
        :param labels: List of numpy arrays, each representing a sequence of labels.
        :param transform: Transformations to be applied to each image
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return the total number of sequences in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch the image sequence and label sequence at the specified index.
        :param idx: Index of the data point to fetch.
        :return: A tuple containing the image sequence and label sequence.
        """
        image_sequence = self.images[idx]
        label_sequence = self.labels[idx]
        
        # Process each image in the sequence
        if self.transform:
            image_sequence = [self.transform(torch.from_numpy(img).float().unsqueeze(0)) for img in image_sequence]
        else:
            image_sequence = [torch.from_numpy(img).float().unsqueeze(0) for img in image_sequence]
        
        # Convert to a single tensor
        image_sequence = torch.stack(image_sequence)
        label_sequence = torch.tensor(label_sequence, dtype=torch.long)
        
        return image_sequence, label_sequence


def load_image(img_path):
    """
    Load and preprocess a single image.
    :param img_path: Path to the image file.
    :return: Preprocessed image as a numpy array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        return img / 255.0
    return None

def load_images_and_labels(data_dir,path_data):
    """
    Load and preprocess all images and labels from the dataset.
    :param path_data: Path to the CSV file containing dataset metadata.
    :return: Lists containing arrays of images and labels for each sequence.
    """
    
    df = pd.read_csv(path_data)
    df['image_path'] = df.apply(lambda row: os.path.join(data_dir, row['scan_name']), axis=1)
    df = df[df['image_path'].apply(os.path.exists)]
    df.sort_values(by=['patient_id','folder_name', 'scan_number'], inplace=True)
    
    X = [] # image sequence
    Y = [] # labels
    
    appointments = df['folder_name'].unique()

    with ThreadPoolExecutor(max_workers=8) as executor:
        for appointment in appointments:
            slices = df[df['folder_name'] == appointment]
            future_to_image = {executor.submit(load_image, row['image_path']): row['sequence_label'] for _, row in slices.iterrows()}
            
            images = []
            labels = []
            for future in future_to_image:
                img = future.result()
                if img is not None:
                    images.append(img)
                    labels.append(future_to_image[future])
            
            if len(images) == 500:
                X.append(np.array(images))
                Y.append(np.array(labels))
    
    return np.array(X), np.array(Y)

def visualize_data(data_loader):
    # Fetch the first batch from the DataLoader
    images, labels = next(iter(data_loader))
    
    # Visualize the first few images from the first sequence in the batch
    sequence_images, sequence_labels = images[0], labels[0]
    
    plt.figure(figsize=(15, 5))
    for i in range(min(len(sequence_images), 5)):  # Show the first 5 images of the sequence
        plt.subplot(1, 5, i + 1)
        # Convert image tensor to numpy for visualization
        img = sequence_images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {sequence_labels[i].item()}')
        plt.axis('off')
    plt.show()

def check_shapes(data_loader):
    # Get the first batch from the DataLoader
    images, labels = next(iter(data_loader))
    print("Shape of images batch:", images.shape)
    print("Shape of labels batch:", labels.shape)

def data_loader(data_dir,path_data, batch_size=16):
    # Load the data
    X, Y = load_images_and_labels(data_dir,path_data)

    # Convert the data to tensors
    X = [np.stack(sequence) for sequence in X]  # Stack images in each sequence
    Y = [np.array(sequence) for sequence in Y]  # Convert labels to arrays

    # Split the data into train, validation, and test sets (60/20/20 split)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Define any additional transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Create dataset objects
    train_dataset = EyeScanDataset(X_train, Y_train, transform=transform)
    val_dataset = EyeScanDataset(X_val, Y_val, transform=transform)
    test_dataset = EyeScanDataset(X_test, Y_test, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # visualize_data(train_loader)
    # check_shapes(train_loader)
    
    return train_loader, val_loader, test_loader
    