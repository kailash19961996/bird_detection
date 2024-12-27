# camera fix 2

import cv2
import numpy as np
from picamera2 import Picamera2
import time

def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    return picam2

def close_camera(picam2):
    if picam2:
        picam2.stop()

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
                frame = picam2.capture_array()  # if this fails, go to except block
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if first_frame is None:
                    first_frame = gray
                    time_to_capture_next += 5
                    continue

                frame_delta = cv2.absdiff(first_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                movement_detected = any(cv2.contourArea(c) >= 500 for c in contours)
                if movement_detected:
                    print("Movement detected!")

                first_frame = gray
                time_to_capture_next += 5

            # Only show if we have a valid frame
            if frame is not None:
                cv2.imshow("Live", frame)
            if thresh is not None:
                cv2.imshow("Thresh", thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error encountered: {e}")
    finally:
        close_camera(picam2)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()