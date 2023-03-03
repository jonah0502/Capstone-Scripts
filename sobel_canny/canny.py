import cv2
import os

# Define input and output directories
input_dir = "welds"
output_dir = "canny_out"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define Canny edge detection parameters
canny_threshold1 = 150
canny_threshold2 = 300

# Loop over images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        filepath = os.path.join(input_dir, filename)
        img = cv2.imread(filepath)

        # Apply Canny edge detection
        edges = cv2.Canny(img, canny_threshold1, canny_threshold2)

        # Save output image
        outpath = os.path.join(output_dir, filename)
        cv2.imwrite(outpath, edges)
