import torch
import torch.nn as nn
import cv2
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 if input is 128x128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # 64 channels, each 16x16
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)  # final output: 2 classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = SimpleCNN()
model.load_state_dict(torch.load("bird_classifier.pth", map_location=torch.device('cpu')))
model.eval()

def classify_image(frame):
    # 1. Convert to the same resolution and transforms as training
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # normalize 0-1
    img = (img - 0.5) / 0.5  # if you used mean=[0.5]*3, std=[0.5]*3
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    
    # If predicted == 0 => bird, if 1 => not_bird, or vice versa
    return predicted.item()