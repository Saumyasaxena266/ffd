# main.py

import cv2
from detection import detect_fruits
from classifier import classify_freshness
from camera import get_webcam_feed
from utils import draw_label, draw_freshness_index

def main():
    # Start capturing from the webcam
    for frame in get_webcam_feed():
        # Detect fruits in the frame
        boxes, scores, classes = detect_fruits(frame)
        
        # Process detected fruits
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                ymin, xmin, ymax, xmax = boxes[i]
                h, w, _ = frame.shape
                box = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
                
                # Crop the detected fruit
                startX, startY, endX, endY = box
                roi = frame[startY:endY, startX:endX]
                
                # Classify fruit freshness (get freshness index)
                freshness_index = classify_freshness(roi)
                
                # Draw bounding box (optional label)
                # You can pass an empty label or customize it as needed
                draw_label(frame, "", box, (0, 255, 0) if freshness_index > 50 else (0, 0, 255))
                
                # Draw freshness index
                freshness_text = f'Freshness: {freshness_index:.2f}%'
                draw_freshness_index(frame, freshness_text, box, (0, 255, 0) if freshness_index > 50 else (0, 0, 255))
        
        # Show the output frame
        cv2.imshow('Fruit Detection and Freshness Index', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
