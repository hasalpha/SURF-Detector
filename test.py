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
        assert isfile(args[i]), f"{i+1}th arg does not exist"

        # Ensures format compliance of image files
        assert isValid(
            args[i].lower(), EXTS), f"{i+1}th arg should be one of {'|'.join(EXTS)}."

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
    img_name = image
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
    print(f"# of keypoints in {img_name} is {len(train_keypoints)}")

    # Display the original image and keypoint drawn image
    cv2.imshow('Output', get_stitched_horizontal_images(
        image, cross_plotted_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    range = max - min
    return [(a - min) / range for a in x]


def task2(images):
    dictionarySize = len(images)
    # Create the SURF detector with minimum hessian threshold set to 1000
    surf = cv2.xfeatures2d.SURF_create(1000)
    total = 0
    descriptions = []
    image_array = []
    # K size 5% 10% 15%
    kSizes = [5, 10, 15]
    for image in images:
        img_name = image
        image = cv2.imread(image)
        image = image_resize_VGA(image)
        rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yCrCbimage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        luminanceImage = yCrCbimage[:, :, 0]
        # Compute surf descriptor and keypoints for Y Channel image
        kp, dsc = surf.detectAndCompute(luminanceImage, None)
        image_array.append((rgbimage, dsc))
        print('# of keypoints in {} is {}'.format(img_name, len(kp)))
        descriptions.append(dsc)
        total += len(kp)

    for k in kSizes:
        KSize = int(k / 100 * total)
        print('K = {}% of {} = {}'.format(k, total, KSize))
        print('Dissimilarity Matrix')
        # create BOW from Kmeans (k size)
        BOW = cv2.BOWKMeansTrainer(KSize)
        for dsc in descriptions:
            BOW.add(dsc)
        vocabulary = BOW.cluster()
        matcher = cv2.FlannBasedMatcher_create()
        matcher.add(vocabulary)
        matcher.train()
        # calculate histrogram from image_array
        histograms = []
        for img in image_array:
            result = np.zeros((KSize, 1), np.float32)
            matches = matcher.match(np.float32(img[1]), vocabulary)
            for match in matches:
                visual_word = match.trainIdx
                result[visual_word] += 1
            histograms.append(result.ravel())
        # Normalize histogram
        normalizedHistogram = normalize(np.array(histograms))
        table = []
        # Comparing two histograms using Chi-Square
        for i in normalizedHistogram:
            list = []
            for j in normalizedHistogram:
                a = cv2.compareHist(i, j, cv2.HISTCMP_CHISQR)
                list.append(a)
            table.append(list)
        # Print the result
        format_row = "{:>20}" * (len(images) + 1)
        print(format_row.format("", *images))
        for team, row in zip(images, table):
            print(format_row.format(team, *row))


if __name__ == '__main__':
    args = sys.argv[1:]
    if validateArgs(args):
        task1(args[0])
    else:
        task2(args[:])