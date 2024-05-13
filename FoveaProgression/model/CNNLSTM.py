import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class CNNLSTMSeq2Seq(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNLSTMSeq2Seq, self).__init__()
        
        # Use a pre-trained ResNet model, modifying it for grayscale input
        resnet = models.resnet18(pretrained=True)
        # Adjust the first convolutional layer to accept 1 channel input (grayscale images)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the fully connected and average pooling layers from ResNet
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Example to determine the output feature size dynamically
        # Temporarily pass a dummy data through the feature extractor
        with torch.no_grad():
            dummy_data = torch.zeros(1, 1, 128, 128)  # Assuming input images are 128x128
            dummy_features = self.feature_extractor(dummy_data)
            # Now get the total features produced by the CNN part
            cnn_output_features = dummy_features.numel() // dummy_features.shape[0]  # Normalize by batch size

        # LSTM part for sequence processing with bidirectional support
        self.lstm = nn.LSTM(input_size=cnn_output_features,  # Adjust based on the output size of ResNet dynamically
                            hidden_size=128,  # Hidden size for LSTM layer
                            num_layers=2,  # Number of LSTM layers
                            batch_first=True,  # Input/Output tensors are provided as (batch, seq, feature)
                            bidirectional=True,  # Bidirectional LSTM
                            dropout=0.5)  # Dropout for regularization
        
        # Fully connected layer to produce a prediction for each time step
        self.fc = nn.Linear(128 * 2, num_classes)  # Multiply by 2 for bidirectional LSTM
        
    def forward(self, x):
        # x shape is (batch, sequence, channel, height, width)
        batch_size, sequence_length, C, H, W = x.size()
        
        # Process each image through the CNN feature extractor
        x = x.view(batch_size * sequence_length, C, H, W)
        cnn_out = self.feature_extractor(x)
        
        # Flatten the CNN output to feed into the LSTM
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = cnn_out.view(batch_size, sequence_length, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)  # lstm_out shape: (batch, seq_length, hidden_size * 2)
        
        # Fully connected layer for each time step
        out = self.fc(lstm_out)  # out shape: (batch, seq_length, num_classes)
        
        return out

class CNNLSTMTest(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNLSTMTest, self).__init__()
        
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