import cv2
import numpy as np
from PIL import Image


import io
import json
import cv2
import os


def scan(binaryimg):
    data = {"success": False}
    if binaryimg is None:
        return data

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    colored_edges = detect_and_draw_edges(image, 100, 200, (255, 255, 0), 1)
    transparent_image = convert_to_transparent_png(colored_edges)
    # return the data dictionary as a JSON response
    return transparent_image

def read_cv2_image(binaryimg):

    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def detect_and_draw_edges(image_path, threshold1, threshold2, line_color, line_thickness):
    """
    Detect edges in an image and draw them on the original image using the specified color and thickness.

    :param image_path: Path to the input image.
    :param threshold1: First threshold for the hysteresis procedure.
    :param threshold2: Second threshold for the hysteresis procedure.
    :param line_color: Color of the edge lines (B, G, R) format.
    :param line_thickness: Thickness of the edge lines.
    :return: Image with detected edges drawn.
    """
    # Read the image
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image=image_path
    blank = np.zeros(image.shape, dtype='uint8')
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(image,(3,3), sigmaX=0, sigmaY=0) 
    med_blur = cv2.medianBlur(image,7)
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detector
    edges = cv2.Canny(img_blur, threshold1, threshold2)

    # Find contours based on edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(blank, contours, -1, line_color, line_thickness)

    return blank

def convert_to_transparent_png(image):
    """
    Convert the image with drawn edges to a PNG with a transparent background,
    keeping only the contours.

    :param image: Image with drawn edges.
    :return: Image with transparent background and contours.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask of the contours
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Create an alpha channel with the mask
    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

    # Merge the alpha channel with the original image
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    return dst

def resize_indexed_color_image_opencv(image_path, output_path, new_width, new_height):
    """
    Resize an indexed color image using OpenCV, attempting to preserve the color palette.

    :param image_path: Path to the input indexed color image.
    :param output_path: Path to save the resized indexed color image.
    :param new_width: New width for the resized image.
    :param new_height: New height for the resized image.
    """
    # Load the image with Pillow to access the palette
    pil_image = Image.open(image_path)

    # Ensure the image is in 'P' mode (indexed color)
    if pil_image.mode != 'P':
        raise ValueError("The image is not in indexed color mode.")

    # Extract the color palette
    palette = pil_image.getpalette()

    # Convert the PIL image to a full-color image
    full_color_image = pil_image.convert("RGB")

    # Convert the PIL image to an OpenCV image
    open_cv_image = np.array(full_color_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Resize the image using OpenCV
    resized_image = cv2.resize(open_cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert the resized image back to PIL format
    resized_pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Re-apply the original palette
    resized_pil_image.putpalette(palette)
    resized_pil_image = resized_pil_image.convert("P")

    # Save the resized image
    resized_pil_image.save(output_path)

# Example usage
#resize_indexed_color_image_opencv('./3.bmp', './resized_image.png', 800, 600)

# Example usage
#colored_edges = detect_and_draw_edges('a.jpg', 100, 200, (255, 255, 0), 1)
#cv2.imwrite('./output.png', colored_edges)
#transparent_image = convert_to_transparent_png(colored_edges)
#cv2.imwrite('./output2.png', transparent_image)
