# Cuts largest contour from an image or folder of images
import cv2
from PIL import Image
import numpy as np
import os

def cut_largest_contour(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    # Bitwise AND
    result = cv2.bitwise_and(image, image, mask=mask)
    # crop
    x, y, w, h = cv2.boundingRect(largest_contour)
    result = result[y:y+h, x:x+w]
    # Convert to PIL image
    result = Image.fromarray(result)
    # Save image
    result.save(output_path)
    
INPUT_PATH = '06'
OUTPUT_PATH = 'padlock_cut'

# check if input is a folder
if os.path.isdir(INPUT_PATH):
    # create output folder
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # loop through all images in the folder
    for filename in os.listdir(INPUT_PATH):
        input_image_path = os.path.join(INPUT_PATH, filename)
        output_image_path = os.path.join(OUTPUT_PATH, filename)
        cut_largest_contour(input_image_path, output_image_path)
else:
    cut_largest_contour(INPUT_PATH, OUTPUT_PATH)

print('Done!')