# camera.py

import cv2

def get_webcam_feed():
    """Capture video from webcam and yield frames."""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        yield frame

    cap.release()
    cv2.destroyAllWindows()
