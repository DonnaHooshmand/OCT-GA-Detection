import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from dataloader import *
from CNNLSTM import *
from train import *
from evaluate import*
from log import *

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():

    experiment_dir = create_experiment_folders('./FoveaProgression/experiments')
    
    num_classes=3 #number of classes
    # model = CNNLSTMTest(num_classes)
    model = CNNLSTMSeq2Seq(num_classes)
    
    # model parameters
    lr = 0.0001 #learning rate
    num_epochs = 100 # training epochs
    batch_size = 1 #bath size
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 5, 10])) # Loss function
    optimizer = optim.Adam(model.parameters(), lr) # Optimizer
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print("Model is using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Model is using CPU")
    print(model)
    save_model_architecture(model, os.path.join(experiment_dir, 'model_architecture', 'model_architecture.txt'))
    
    data_path = './FoveaProgression/data/experiment/X/'
    data_path = './FoveaProgression/data/sample/'
    train_path = data_path + 'train.csv'
    val_path = data_path + 'val.csv'
    test_path = data_path + 'test.csv'
    
    # data_dir = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3'
    user_id = os.getuid()
    data_dir = f"/run/user/{user_id}/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3"

    if not os.path.exists(data_dir):
        logging.error("\nData directory does not exist.")
        sys.exit(1)
    else:
        print("\nDirectory found, loading data ...")
    
    # train_loader, val_loader, test_loader = data_loader(data_dir,data_path,batch_size) 
    train_loader = data_loader(data_dir,train_path,batch_size)
    print("\nTrain Loader complete")
    val_loader = data_loader(data_dir,val_path,batch_size)
    print("\nValidation Loader complete")
    test_loader = data_loader(data_dir,test_path,batch_size)
    print("\nTest Loader complete")
    log_data_details(train_loader, val_loader, test_loader, os.path.join(experiment_dir, 'train_val_test_details', 'data_details.txt'))

    best_model_path = os.path.join(experiment_dir, 'best_model', 'best_model.pth')
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path, experiment_dir)

    torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_weights', 'final_model_weights.pth'))

    evaluate_and_save_results(model, test_loader, experiment_dir)

    # # Git commit and push changes
    # base_branch = "training"
    # timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # new_branch = f"training-/{timestamp}"
    # commit_message = f"Training completed on {timestamp}"
    # git_commit_and_push_changes(base_branch, new_branch, commit_message)
    
if __name__ == "__main__":
    main()
    