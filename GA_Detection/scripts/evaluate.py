from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import torch
import csv
import os
import numpy as np
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def evaluate_and_save_results(model, test_loader, experiment_dir):
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for images, labels, folder_names in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = model(images)
            batch_size, seq_len, num_classes = outputs.size()
            
            # Flatten the outputs and labels for evaluation
            outputs = outputs.view(batch_size, seq_len, num_classes)
            labels = labels.view(batch_size, seq_len)
            
            for seq_idx in range(batch_size):
                seq_outputs = outputs[seq_idx]
                seq_labels = labels[seq_idx]
                
                _, predicted = torch.max(seq_outputs, 1)
                
                seq_true_labels = seq_labels.cpu().numpy()
                seq_pred_labels = predicted.cpu().numpy()
                
                # Save predictions and labels for each sequence to a CSV file
                folder_name = folder_names[seq_idx]
                output_csv_path = os.path.join(experiment_dir, 'results', f'{folder_name}.csv')
                with open(output_csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Frame Index', 'True Label', 'Predicted Label'])
                    for frame_idx, (true_label, pred_label) in enumerate(zip(seq_true_labels, seq_pred_labels)):
                        writer.writerow([frame_idx, true_label, pred_label])
            
            ## Grad Cam 
                    
            target_layers = [model.resnet.layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
                    
            targets = [ClassifierOutputTarget(2)]
            breakpoint()
                
            ## create an input tensor image for your model
            ## input_tenso can be a batch tensor with several images
            images = images.requires_grad_(True)
            grayscale_cam = cam(images, targets=targets)
                    
            # visualization = show_cam_on_image(images, grayscale_cam, use_rgb=False)
                    
            model.outputs = cam.outputs
                    
            # im = Image.fromarray(visualization)
            cv2.imwrite(os.path.join(experiment_dir, 'test_picture_outputs', f'{folder_name+seq_idx}.jpg'), grayscale_cam[0,:,:])
                
                
                
    
    
    all_labels, all_predictions = aggregate_all_labels_predictions(test_loader, model)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    
    # Plot and save confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, os.path.join(experiment_dir, 'confusion_matrix', 'confusion_matrix.png'))

def aggregate_all_labels_predictions(test_loader, model):
    all_labels = []
    all_predictions = []
    
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten the output for evaluation
            labels = labels.view(-1)

            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_predictions

def visualize_predictions(data_loader, model, num_sequences=1):
    model.eval()
    images, labels = next(iter(data_loader))
    if torch.cuda.is_available():
        images = images.cuda()
    
    with torch.no_grad():
        outputs = model(images)
    outputs = outputs.view(-1, outputs.size(-1))
    _, predicted = torch.max(outputs, 1)
    
    images = images.cpu()
    labels = labels.view(-1).cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    for i in range(min(len(images[0]), 5)):  # Visualize first 5 images of the first sequence
        plt.subplot(1, 5, i + 1)
        img = images[0][i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predicted[i]}, True: {labels[i]}')
        plt.axis('off')
    plt.show()

def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'loss_accuracy_curves', 'loss_curve.png'))

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'loss_accuracy_curves', 'accuracy_curve.png'))

def plot_confusion_matrix(true_labels, pred_labels, output_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(set(true_labels)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
