import torch
import torch.nn as nn
import torch.optim as optim

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNLSTM, self).__init__()
        
        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input is grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # LSTM part
        self.lstm = nn.LSTM(input_size=64 * 16 * 16,  
                            hidden_size=128,         # Hidden size for LSTM layer
                            num_layers=1,            # Number of LSTM layers
                            batch_first=True)        # Input/Output tensors are provided as (batch, seq, feature)
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)         # Output size is the number of classes
        
    def forward(self, x):
        # x shape is (batch, sequence, channel, height, width)
        batch_size, sequence_length, C, H, W = x.size()
        
        # Reshape to work with images individually (batch * sequence, channel, height, width)
        x = x.view(batch_size * sequence_length, C, H, W)
        
        # CNN operations
        x = self.cnn(x)
        
        # Flatten the output for LSTM
        x = x.view(batch_size, sequence_length, -1)
        
        # LSTM operations
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_length, hidden_size)
        
        # Apply the fully connected layer to each time step
        x = self.fc(lstm_out)  # x shape: (batch, seq_length, num_classes)
        
        return x