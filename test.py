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
    image1 = cv2.imread(image)
    image1 = image_resize(image1)
    print(image1.shape)
    # # Convert the training image to RGB
    # training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # # Convert the training image to gray scale
    # training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

    # surf = cv2.xfeatures2d.SURF_create(1000)
    # surf.setExtended(True)
    # train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
    # pts = cv2.KeyPoint_convert(train_keypoints)
    # for i in pts:
    #     x,y = i.astype(int)
    #     cv2.line(training_image, (x-5, y), (x+5, y), 0, thickness=1)
    #     cv2.line(training_image, (x, y-5), (x, y+5), 0, thickness=1)
    # # keypoints_without_size = np.copy(training_image)
    # keypoints_with_size = np.copy(training_image)
    # # cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
    # cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # keypoints_with_size = cv2.cvtColor(keypoints_with_size, cv2.COLOR_RGB2BGR)
    # print(len(train_keypoints))
    # cv2.imshow('Output', keypoints_with_size)
    # # training_image = cv2.cvtColor(training_image, cv2.COLOR_RGB2BGR)
    # # cv2.imshow('Output', training_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    args = sys.argv[1:]
    if validateArgs(args):
        task1(args[0])
    else:
        print('task2')
