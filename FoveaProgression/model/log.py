from datetime import datetime
import os

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