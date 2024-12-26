
print ("checking")

import cv2
import numpy as np
from picamera2 import Picamera2
import time

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()

    first_frame = None

    while True:
        # Capture current frame as a NumPy array
        frame = picam2.capture_array()

        # Convert to grayscale & blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            continue

        # Frame difference
        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            movement_detected = True
            break

        if movement_detected:
            print("Movement detected!")
            # Optionally capture a still (to file) or pass `frame` to your AI model

        # Show windows (for debugging)
        cv2.imshow("Live", frame)
        cv2.imshow("Thresh", thresh)

        # Update the reference frame
        first_frame = gray

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()