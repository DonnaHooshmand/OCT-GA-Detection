import torch
from evaluate import*
from log import*

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path, experiment_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            # Forward pass
            outputs = model(images)
            batch_size, sequence_length, num_classes = outputs.shape

            # Reshape outputs and labels for CrossEntropyLoss
            outputs = outputs.view(batch_size * sequence_length, num_classes)
            labels = labels.view(batch_size * sequence_length)

            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate average loss and accuracy over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validate the model
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = model(images)
                batch_size, sequence_length, num_classes = outputs.shape

                outputs = outputs.view(batch_size * sequence_length, num_classes)
                labels = labels.view(batch_size * sequence_length)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the best model and commit changes to git
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # Plot and save loss and accuracy curves
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir)

    # Save training and validation losses and accuracies to a file
    save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir)


def train_model_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path, experiment_dir, patience=15):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            # Forward pass
            outputs = model(images)
            batch_size, sequence_length, num_classes = outputs.shape

            # Reshape outputs and labels for CrossEntropyLoss
            outputs = outputs.view(batch_size * sequence_length, num_classes)
            labels = labels.view(batch_size * sequence_length)

            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validate the model
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = model(images)
                batch_size, sequence_length, num_classes = outputs.shape

                outputs = outputs.view(batch_size * sequence_length, num_classes)
                labels = labels.view(batch_size * sequence_length)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Plot and save loss and accuracy curves
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir)

    # Save training and validation losses and accuracies to a file
    save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir)

def calculate_class_weights(train_loader, num_classes):
    """Calculate class weights based on the frequency of each class in the training dataset."""
    class_counts = np.zeros(num_classes)
    for _, labels, _ in train_loader:
        labels = labels.view(-1)
        for label in labels:
            class_counts[label.item()] += 1
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.FloatTensor(class_weights)