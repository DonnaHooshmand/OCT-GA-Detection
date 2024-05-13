import torch
from evaluate import*

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path, experiment_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0

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
        
        # Calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        correct = 0
        total = 0
        
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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # Plot and save loss and accuracy curves
    plot_curves(train_losses, val_losses, val_accuracies, experiment_dir)