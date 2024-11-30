import os
import tensorflow as tf
import cv2
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the detection model
model_dir = r'C:\Users\saxen\OneDrive\Desktop\New folder\ssd_mobilenet_v2_320x320_coco17_tpu-8'
detection_model = tf.saved_model.load(str(model_dir))

def run_inference(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def save_detections(image, detections, output_path):
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    threshold = 0.5  # Confidence threshold
    height, width, _ = image.shape
    detected = False  # Flag to check if any objects are detected

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            detected = True  # Set flag if an object is detected

    # Save the image with detections if any objects were detected
    if detected:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved detected image: {output_path}")

# Process all images in the dataset folder
dataset_path = r'C:\Users\saxen\OneDrive\Desktop\New folder\dataset'
output_dir = r'C:\Users\saxen\OneDrive\Desktop\New folder\output_images'  # Output directory for detected images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of valid image extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Loop through the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        # Check if the file is an image
        if file.lower().endswith(valid_extensions):
            image_path = os.path.join(root, file)
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Convert image from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare the output file path
            output_image_path = os.path.join(output_dir, file)
            
            # Run detection
            detections = run_inference(detection_model, image_rgb)
            
            # Save the detections directly without displaying
            save_detections(image_rgb, detections, output_image_path)


