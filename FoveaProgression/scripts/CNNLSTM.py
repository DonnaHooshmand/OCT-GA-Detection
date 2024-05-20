import torch
import torch.nn as nn
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