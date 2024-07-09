from datetime import datetime
import os
import sys
import logging
import subprocess


def save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, experiment_dir):
    metrics_path = os.path.join(experiment_dir, 'loss_accuracy_curves', 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Epoch\tTrain Loss\tValidation Loss\tTrain Accuracy\tValidation Accuracy\n')
        for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(zip(train_losses, val_losses, train_accuracies, val_accuracies), start=1):
            f.write(f'{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{train_acc:.4f}\t{val_acc:.4f}\n')

def git_commit_and_push_changes(base_branch, new_branch, commit_message):
    """Commit all changes to git with the provided commit message and push to a new branch."""
    try:
        # Ensure we are on the base branch
        subprocess.run(["git", "checkout", base_branch], check=True)
        
        # Create and switch to the new branch
        subprocess.run(["git", "checkout", "-b", new_branch], check=True)
        
        # Add, commit, and push changes
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", "origin", new_branch], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error committing and pushing changes to git: {e}")
        sys.exit(1)

def create_experiment_folders(base_dir):
    """Create directories for the experiment logs."""
    date_str = datetime.now().strftime('%Y%m%d')
    time_str = datetime.now().strftime('%H%M%S')
    experiment_dir = os.path.join(base_dir, date_str, time_str)
    os.makedirs(experiment_dir, exist_ok=True)
    subdirs = ['model_architecture', 'train_val_test_details', 'best_model', 'model_weights', 'loss_accuracy_curves', 'confusion_matrix', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir

def save_model_architecture(model, path):
    with open(path, 'w') as f:
        f.write(str(model))

def log_data_details(train_loader, val_loader, test_loader, path):
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    with open(path, 'w') as f:
        f.write(f'Training Set Size: {train_size}\n')
        train_appointments = [folder_name for _, _, folder_name in train_loader.dataset]
        f.write(f'Training Appointments: {train_appointments}\n')
        
        f.write(f'\nValidation Set Size: {val_size}\n')
        val_appointments = [folder_name for _, _, folder_name in val_loader.dataset]
        f.write(f'Validation Appointments: {val_appointments}\n')

        f.write(f'\nTest Set Size: {test_size}\n')
        test_appointments = [folder_name for _, _, folder_name in test_loader.dataset]
        f.write(f'Test Appointments: {test_appointments}\n')