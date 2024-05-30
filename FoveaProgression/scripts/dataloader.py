import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle

class CLAHETransform:
    def __init__(self, clip_limit=1.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img = img.numpy().squeeze(0)  # Convert tensor to numpy array and remove the channel dimension
        img_clahe = self.clahe.apply(np.uint8(img * 255))  # Apply CLAHE and scale back to [0, 255]
        img_clahe = torch.from_numpy(img_clahe / 255.0).unsqueeze(0).float()  # Convert back to tensor
        return img_clahe

class EyeScanDataset(Dataset):
    def __init__(self, images, labels, folder_name, transform=None):
        """
        Initialize the dataset.
        :param images: List of numpy arrays, each representing a sequence of images.
        :param labels: List of numpy arrays, each representing a sequence of labels.
        :param transform: Transformations to be applied to each image
        """
        self.images = images
        self.labels = labels
        self.scan_id = folder_name
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
        scan_id = self.scan_id[idx]

        
        # Process each image in the sequence
        if self.transform:
            image_sequence = [self.transform(torch.from_numpy(img).float().unsqueeze(0)) for img in image_sequence]
        else:
            image_sequence = [torch.from_numpy(img).float().unsqueeze(0) for img in image_sequence]

        # Convert to a single tensor
        image_sequence = torch.stack(image_sequence)
        label_sequence = torch.tensor(label_sequence, dtype=torch.long)
        
        return image_sequence, label_sequence, scan_id


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
    scan_id = []  # appointment scan folder names

    
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
                scan_id.append(appointment)

    
    return np.array(X), np.array(Y), scan_id

def visualize_data(data_loader):
    # Fetch the first batch from the DataLoader
    images, labels, _ = next(iter(data_loader))
    
    # Denormalize images
    images = images * 0.229 + 0.485
    
    # Iterate over the batch and find images of class 2
    class_2_images = []
    class_2_labels = []
    
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if labels[i][j] == 2:
                class_2_images.append(images[i][j])
                class_2_labels.append(labels[i][j])
                if len(class_2_images) >= 5:  # Limit to the first 5 images
                    break
        if len(class_2_images) >= 5:
            break
    
    if class_2_images:
        plt.figure(figsize=(15, 5))
        for i in range(len(class_2_images)):  # Show the first 5 images of class 2
            plt.subplot(1, 5, i + 1)
            # Convert image tensor to numpy for visualization
            img = class_2_images[i].squeeze().numpy()
            plt.imshow(img, cmap='gray')
            plt.title(f'Label: {class_2_labels[i].item()}')
            plt.axis('off')
        plt.show()
    else:
        print("No images of class 2 found in the batch.")

def check_shapes(data_loader):
    # Get the first batch from the DataLoader
    images, labels, _ = next(iter(data_loader))
    print("Shape of images batch:", images.shape)
    print("Shape of labels batch:", labels.shape)

def save_cached_data(dataset, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)

def load_cached_data(cache_file):
    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def data_loader(data_dir, path_data, batch_size, cache_file):
    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        dataset = load_cached_data(cache_file)
    else:
        print("Processing data...")
        X, Y, scan_id = load_images_and_labels(data_dir, path_data)

        # Define the transformations
        transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Convert the data to tensors
        X = [np.stack(sequence) for sequence in X]  # Stack images in each sequence
        Y = [np.array(sequence) for sequence in Y]  # Convert labels to arrays

        dataset = EyeScanDataset(X, Y, scan_id, transform=transform)  # Create dataset objects
        # save_cached_data(dataset, cache_file)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoaders
    # visualize_data(data_loader)
    return data_loader