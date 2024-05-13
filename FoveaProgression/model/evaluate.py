from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
import csv


def evaluate_and_save_results(model, test_loader, output_csv_path):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten the output for evaluation
            labels = labels.view(-1)

            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    # Save predictions and labels to a CSV file
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Index', 'True Label', 'Predicted Label'])
        for i, (true_label, pred_label) in enumerate(zip(all_labels, all_predictions)):
            writer.writerow([i, true_label, pred_label])

def visualize_filtered_predictions(data_loader, model, target_labels=[1, 3, 5]):
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
    
    # Filter indices based on target labels
    target_indices = [i for i, label in enumerate(labels) if label in target_labels]
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(target_indices[:5]):  # Show up to 5 images meeting the criteria
        plt.subplot(1, len(target_indices[:5]), i + 1)
        img = images[0][idx].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predicted[idx]}, True: {labels[idx]}')
        plt.axis('off')
    plt.show()


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
