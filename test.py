import cv2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from os.path import isfile

from numpy.lib.function_base import disp

# Image file extensions recognized by opencv
EXTS = ['.jpeg', '.jpg', '.jpe', '.bmp', '.dib', '.jp2',
        '.png', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm',
        '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr',
        '.hdr', '.pic']

# Color spaces options as per assignment specifications
COLOR_SPACES = ['-XYZ', '-Lab', '-YCrCb', '-HSB']

# Checks to see if provided arguments are of valid format
def isValid(string, options):
    # Returns true if there is atleast one match in the options list
    return any(extension in string for extension in options)

# Comprehensive input checks to invalidate any erroneous arguments passed to the program
def validateArgs(args):

    number_of_images = len(args)

    # Ensures atleast one argument is passed
    assert number_of_images >= 1, f"Please provide atleast one argument"

    for i in range(number_of_images):

        # Ensures if the passed 1st argument exists or not
        assert isfile(args[i]), f"{i}th arg does not exist"

        # Ensures format compliance of image files
        assert isValid(
            args[i].lower(), EXTS), f"{i}th arg should be one of {'|'.join(EXTS)}."

    if number_of_images == 1:
        return True

    return False


def get_stitched_horizontal_images(original_img, c1):

    # Performing two horizontal concatenations and finally concatenating the two outputs
    img_c1 = cv2.hconcat([original_img, c1])

    # Resizing the final image to fit the display window
    return img_c1


def display_image(img):
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output', 1280, 720)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_resize_VGA(img):
    height, width = img.shape[:2]
    max_height = 480
    max_width = 600

    # Only shrink if image is bigger than maximum dimensions
    if max_height < height or max_width < width:

        # Determine scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)

        # Resize image
        img = cv2.resize(img, None, fx=scaling_factor,
                         fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def plot_cross(pts, image):

    # Convert keypoints into numpy arrays
    pts = cv2.KeyPoint_convert(pts)

    # Extract coordinates from the numpy arrays and plot cross at the location
    for i in pts:
        x, y = i.astype(int)
        cv2.line(image, (x-5, y), (x+5, y), 0, thickness=1)
        cv2.line(image, (x, y-5), (x, y+5), 0, thickness=1)
    return image


def task1(image):

    # Read in source image
    image = cv2.imread(image)

    # Resize the source image to a VGA comparable format
    image = image_resize_VGA(image)

    # Convert the training image to YCrCb
    training_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Get the Y component of the image corresponding to luminance
    luminance_component = training_image[:, :, 0]

    # Create the SURF detector with minimum hessian threshold set to 1000
    surf = cv2.xfeatures2d.SURF_create(800)

    # Use 128-Dimensional computation for SURF
    surf.setExtended(True)

    # Detect and compute keypoints and descriptors from the luminance Component
    train_keypoints, train_descriptor = surf.detectAndCompute(
        luminance_component, None)

    # Plot cross at key points
    cross_plotted_image = plot_cross(train_keypoints, np.copy(image))

    # Draw keypoints with orientation
    cv2.drawKeypoints(cross_plotted_image, train_keypoints, cross_plotted_image,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Print number of keypoints detected in the image
    print(f"# of keypoints detected: {len(train_keypoints)}")

    # Display the original image and keypoint drawn image
    cv2.imshow('Output', get_stitched_horizontal_images(
        image, cross_plotted_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = sys.argv[1:]
    if validateArgs(args):
        task1(args[0])
    else:
        print('task2')
