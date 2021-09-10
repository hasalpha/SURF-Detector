import cv2

from typing import final
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

# Returns cv2.COLOR_XXX code for any given color space.
# Conversions are done using the default BGR channel that Opencv uses while reading an image


def get_code(color_space):
    if color_space == '-XYZ':
        return cv2.COLOR_BGR2XYZ
    elif color_space == '-Lab':
        return cv2.COLOR_BGR2Lab
    elif color_space == '-YCrCb':
        return cv2.COLOR_BGR2YCrCb
    elif color_space == '-HSB':
        return cv2.COLOR_BGR2HSV

# Returns a list of numpy arrays corresponding to Blue, Green and Red channels of the image


def get_converted_color_components(img, code):
    img = cv2.cvtColor(img, code)
    store = []
    for i in [img[:, :, 0], img[:, :, 1], img[:, :, 2]]:
        store += [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)]
    return store

# Returns one image divided into four quadrants and each quadrant represents the images passed in order
def get_stitched_images(original_img, c1, c2, c3):

    # Performing two horizontal concatenations and finally concatenating the two outputs
    img_c1 = cv2.hconcat([original_img, c1])
    c2_c3 = cv2.hconcat([c2, c3])
    img_c1_c2_c3 = cv2.vconcat([img_c1, c2_c3])

    # Resizing the final image to fit the display window
    return image_resize(image=img_c1_c2_c3, width=1280, height=720)

def display_image(img):
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output', 1280, 720)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Makes the source image have same dimensions as destination image


def get_same_shape_image(src, dst):
    width, height = dst.shape[:2]
    src = image_resize(src, width, height)
    return cv2.resize(src, (height, width))

# Returns a Green Screen mask of a high enough range using HSV color values


def get_green_screen_mask_HSV(img):
    return cv2.inRange(img, np.array([50, 100, 80]), np.array([90, 255, 255]))

# Displays a four quadrant image with the 1st quadrant being the original image
# 2nd, 3rd and 4th quadrant correspond to each of the Color channels in the color image
def image_resize(img):
    height, width = img.shape[:2]
    max_height = 480
    max_width = 600

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)

        # resize image
        img = cv2.resize(img, None, fx=scaling_factor,
                         fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def task1(image):

    print(image)
    # Read the image from current directory
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Make a copy of the image
    img_copy = np.copy(img)
    img_copy = image_resize(img_copy)
    #-- Step 1: Detect the keypoints using SURF Detector
    minHessian = 400
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints = detector.detect(img_copy)
    print(len(keypoints))
    #-- Draw keypoints
    img_keypoints = np.empty((img_copy.shape[0], img_copy.shape[1], 3), dtype=np.uint8)
    cv2.drawKeypoints(img_copy, keypoints, img_keypoints)
    #-- Show detected (drawn) keypoints
    cv2.imshow('SURF Keypoints', img_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#     # Get color space conversion code
#     code = get_code(color_space)

#     # Store each color channel arrays into individual variables
#     c1, c2, c3 = get_converted_color_components(img_copy, code)

#     # Stitch the orginal image and corresponding color channels
#     stitched_image = get_stitched_images(img_copy, c1, c2, c3)

#     # Display the final output in a single viewing window
#     display_image(stitched_image)

# # Displays a four quadrant image where 1st quadrant -> Green screen image, 2nd quadrant -> Extracted persona with white background
# # 3rd quadrant -> Scenic image , 4th quadrant -> Chroma Keyed image with a persona on the foreground and the scenic image as background


def task2(green_screen, scenic_image):

    # Read green screen and scenic images
    green_screen_img = cv2.imread(green_screen)
    scenic_img = cv2.imread(scenic_image)

    # Make the green screen image have same dimension as scenic image
    green_screen_img = get_same_shape_image(green_screen_img, scenic_img)

    # Convert the two images to HSV color mapping
    img = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(scenic_img, cv2.COLOR_BGR2HSV)

    # Generate a green screen mask
    mask = get_green_screen_mask_HSV(img)

    # Make green screen background white while preserving the persona
    img[mask != 0] = [0, 0, 255]

    # Make pixels corresponding the persona black to blend the persona
    img2[mask == 0] = [0, 0, 0]

    # Convert the two images back to BGR color space for proper viewing
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

    # Display the final output in a single viewing window
    display_image(get_stitched_images(
        green_screen_img, img, scenic_img, img+img2))


if __name__ == '__main__':
    args = sys.argv[1:]
    if validateArgs(args):
        task1(args[0])
    else:
        print('task2')
