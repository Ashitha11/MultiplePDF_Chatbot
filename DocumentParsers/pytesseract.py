#Extracts all data from an image using pytesseract and OpenCV -->is able to read the contents of the image and print it out

import pytesseract
import cv2  # For image loading

# Point to Tesseract executable (optional if PATH is set)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load an image (replace with your image path)
img = cv2.imread('baby.png')

# Perform OCR
text = pytesseract.image_to_string(img)
print("Extracted Text:", text)
