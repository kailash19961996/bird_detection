import torch
from torchvision import models
from torch.quantization import quantize_dynamic
torch.backends.quantized.engine = 'qnnpack'

model = models.resnet18(pretrained=False)  # or your custom model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # assuming two output classes

# Load your model weights
model.load_state_dict(torch.load('/Users/kailashkumar/Documents/CODE/bird-detector/bird_classifier.pth'))
model.eval()

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

torch.save(quantized_model.state_dict(), 'quantized_model.pth')



