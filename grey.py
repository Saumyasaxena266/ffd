import os
from PIL import Image

# Define the directory containing the images
input_folder = r'C:\Users\DELL\OneDrive\Desktop\RANGE\dataset\Test\freshpatato'  # Replace with your folder path
output_folder = r'C:\Users\DELL\OneDrive\Desktop\RANGE\greyimages\test\freshpatato_grey'  # Replace with your desired output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add other extensions if needed
        # Open an image file
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # Convert the image to grayscale
            grayscale_img = img.convert('L')
            # Save the grayscale image
            grayscale_img.save(os.path.join(output_folder, filename))

print("Conversion to grayscale completed.")



