import cv2
import numpy as np
import time
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from picamera2 import Picamera2
from torch import nn

model = models.resnet18(pretrained=False)  # or your custom model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # assuming two output classes

# Load your model weights
model.load_state_dict(torch.load('quantized_model.pth'))
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class names
class_names = ["bird", "no_bird"]  # Or however your classes are named

########################################
# 2. Helper function to convert frame to model input
########################################
def predict_frame(frame):
    """
    frame: a BGR image (numpy array) from OpenCV
    Returns: predicted class string
    """
    # Convert from BGR (OpenCV) to RGB (PIL) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the numpy array to a PIL Image
    pil_img = Image.fromarray(frame_rgb)
    
    # Apply the same transforms as in training
    input_tensor = test_transforms(pil_img)
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

########################################
# 3. Camera init/close functions
########################################
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    return picam2

def close_camera(picam2):
    if picam2:
        picam2.stop()

########################################
# 4. Main loop: detect motion, run inference if movement_detected
########################################
def main():
    picam2 = init_camera()
    first_frame = None
    time_to_capture_next = time.time() + 5
    frame = None
    thresh = None

    try:
        while True:
            current_time = time.time()
            if current_time >= time_to_capture_next:
                frame = picam2.capture_array()  # capture a new frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if first_frame is None:
                    # Initialize first_frame for motion detection
                    first_frame = gray
                    time_to_capture_next += 5
                    continue

                # Calculate difference
                frame_delta = cv2.absdiff(first_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                # Find contours
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                movement_detected = any(cv2.contourArea(c) >= 500 for c in contours)

                if movement_detected:
                    print("Movement detected!")
                    ########################################
                    # Here’s the new part: call our model
                    ########################################
                    prediction = predict_frame(frame)
                    print(f"Prediction: {prediction}")

                # Update first_frame and time_to_capture_next
                first_frame = gray
                time_to_capture_next += 5

            # Optional: Show windows with live feed and threshold
            if frame is not None:
                cv2.imshow("Live", frame)
            if thresh is not None:
                cv2.imshow("Thresh", thresh)

            # Press 'q' to exit gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")
    except Exception as e:
        print(f"Error encountered: {e}")
    finally:
        close_camera(picam2)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
