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

# Returns two images of same size stitched horizontally


def get_stitched_horizontal_images(original_img, c1):

    # Performing two horizontal concatenations and finally concatenating the two outputs
    img_c1 = cv2.hconcat([original_img, c1])

    # Resizing the final image to fit the display window
    return img_c1


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
    image_name = image
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
    print(f"# of keypoints in {image_name} is {len(train_keypoints)}")

    # Display the original image and keypoint drawn image
    cv2.imshow('Output', get_stitched_horizontal_images(
        image, cross_plotted_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Return normalized histogram as output
def normalize(x):
    min = np.min(x)
    max = np.max(x)
    range = max - min
    return [(a - min) / range for a in x]


def task2(images):
    # Create the SURF detector with minimum hessian threshold set to 1000
    surf = cv2.xfeatures2d.SURF_create(1000)

    # Local variables to create and store image keypoints and descriptors
    total = 0
    descriptions = []
    image_array = []

    # K sizes corresponding to 5%, 10% & 15%
    k_sizes_array = [5, 10, 15]

    # Store descriptors and keypoints of images in aforementioned local variables
    for image in images:

        # Store image name for printing purpose
        image_name = image

        # Read and Resize the image to a VGA comparable format
        image = cv2.imread(image)
        image = image_resize_VGA(image)

        # Convert image from BGR to RGB color space
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image from BGR to YCrCb color space
        yCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Extract the luminance component from the image
        y_component = yCrCb_image[:, :, 0]

        # Compute surf descriptor and keypoints for Y Channel image
        kp, dsc = surf.detectAndCompute(y_component, None)

        # Append the RGB image and its corresponding image descriptor as a tuple to the image array
        image_array.append((rgb_image, dsc))

        # Print number of keypoints present in each image
        print(f'# of keypoints in {image_name} is {len(kp)}')

        # Add the descriptor of current image to the descriptions array
        descriptions.append(dsc)

        # Sum total keypoints
        total += len(kp)

    # Iterate for all K sizes
    for k in k_sizes_array:

        # Determine overall K size based on total keypoints
        k_size = int(k / 100 * total)

        # Create the bag-of-words object from Kmeans (k size)
        BOW = cv2.BOWKMeansTrainer(k_size)

        # Add descriptions into the BOW
        for description in descriptions:
            BOW.add(description)

        # Create the matcher object and train it with the provided vocabulary
        vocabulary = BOW.cluster()
        matcher = cv2.FlannBasedMatcher_create()
        matcher.add(vocabulary)
        matcher.train()

        # Deduce image_histograms from the image_array
        image_histograms = []
        for img in image_array:

            # Create a numpy array of k_size and 1 dimensions and fill it 0.0
            result = np.zeros((k_size, 1), np.float32)

            # Identify all matches based on vocabulary
            matches = matcher.match(np.float32(img[1]), vocabulary)

            # Set all values at determined indices to 1
            for match in matches:
                word = match.trainIdx
                result[word] += 1

            # Append a flattened version of the result array
            image_histograms.append(result.ravel())

        # Normalize the obtained histogram
        normalized_histogram = normalize(np.array(image_histograms))
        grid = []

        # Compare image_histograms using Chi Square comparison method
        for row in normalized_histogram:
            storage = []
            for column in normalized_histogram:
                result = cv2.compareHist(row, column, cv2.HISTCMP_CHISQR)
                storage.append(result)
            grid.append(storage)

        # Printing the output in a human-readable format
        print(f'K = {k}% of {total} = {k_size}')
        print('Dissimilarity Matrix')
        print_row = "{:>20}" * (len(images) + 1)
        print(print_row.format("", *images))
        for team, row in zip(images, grid):
            print(print_row.format(team, *row))


if __name__ == '__main__':
    args = sys.argv[1:]
    if validateArgs(args):
        task1(args[0])
    else:
        task2(args[:])
