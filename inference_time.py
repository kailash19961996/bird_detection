import torch
from torchvision import transforms, models
from PIL import Image
import time

# Define your model architecture (adjust if your model is custom)
model = models.resnet18(pretrained=False)  # Ensure it's the same architecture as used during training
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Adjust the output classes if needed

# Load the state_dict
state_dict = torch.load("bird_classifier.pth")  # Load only the weights
model.load_state_dict(state_dict)
model.eval()

# Dummy input
image_path = "/Users/kailashkumar/Documents/CODE/bird-detector/data/final_checking/_dsc2838.jpg"  # Replace with your test image
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = transform(image).unsqueeze(0)

# Measure inference time
start_time = time.time()
with torch.no_grad():
    output = model(input_tensor)
end_time = time.time()

# Print results
print("Inference Time:", end_time - start_time, "seconds")
print("Predicted Class:", torch.argmax(output).item())