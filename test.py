import cv2

from typing import final
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from os.path import isfile

#Image file extensions recognized by opencv
EXTS = ['.jpeg', '.jpg', '.jpe', '.bmp', '.dib', '.jp2',
              '.png','.pbm', '.pgm', '.ppm', '.pxm', '.pnm',
              '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr',
              '.hdr', '.pic']

#Color spaces options as per assignment specifications
COLOR_SPACES = ['-XYZ', '-Lab', '-YCrCb', '-HSB']

#Checks to see if provided arguments are of valid format
def isValid(string, options):
    #Returns true if there is atleast one match in the options list
    return any(extension in string for extension in options)

#Comprehensive input checks to invalidate any erroneous arguments passed to the program
def validateArgs(args):
    #Ensures only two arguments are passed
    assert len(args) == 2, f"Please provide only two args"
    
    #Ensures if the passed 2nd argument exists or not
    assert isfile(args[1]), f"2nd arg does not exist"
    
    #Ensures format compliance of image files
    assert isValid(args[1], EXTS), f"2nd arg should be one of {'|'.join(EXTS)}."
    
    #Check to see if task corresponds to task-1 or task-2
    if not isValid(args[0], EXTS):
        #If task-1, Checks the color space provided in the first argument
        assert isValid(args[0], COLOR_SPACES), f"1st arg should be one of {'|'.join(COLOR_SPACES)}."
        return True
    #If task-2, Asserts the existence of the first file
    assert isfile(args[0]), f"1st arg does not exist"
   
#Returns cv2.COLOR_XXX code for any given color space.
#Conversions are done using the default BGR channel that Opencv uses while reading an image    
def get_code(color_space):
    if color_space == '-XYZ':
        return cv2.COLOR_BGR2XYZ
    elif color_space == '-Lab':
        return cv2.COLOR_BGR2Lab
    elif color_space == '-YCrCb':
        return cv2.COLOR_BGR2YCrCb
    elif color_space == '-HSB':
        return cv2.COLOR_BGR2HSV

#Returns a list of numpy arrays corresponding to Blue, Green and Red channels of the image
def get_converted_color_components(img, code):
    img = cv2.cvtColor(img, code)
    store = []
    for i in [img[:, :, 0], img[:, :, 1], img[:, :, 2]]:
        store += [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)]
    return store

#Returns a resized image while maintaining aspect ratio
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    #Set current dimensions and height, width of the image
    dimensions = None
    h, w = image.shape[:2]

    #Return same image as width and height are not specified
    if width is None and height is None:
        return image

    if width is None:
        # calculate new width based on provided height
        r = height / float(h)
        dimensions = (int(w * r), height)
    else:
        # calculate new height based on provided width
        r = width / float(w)
        dimensions = (width, int(h * r))

    #Return the resized image
    return cv2.resize(image, dimensions, interpolation = inter) 

#Returns one image divided into four quadrants and each quadrant represents the images passed in order
def get_stitched_images(original_img, c1, c2, c3):
    
    #Performing two horizontal concatenations and finally concatenating the two outputs
    img_c1 = cv2.hconcat([original_img, c1])
    c2_c3 = cv2.hconcat([c2, c3])
    img_c1_c2_c3 = cv2.vconcat([img_c1, c2_c3])
    
    #Resizing the final image to fit the display window
    return image_resize(image=img_c1_c2_c3, width=1280, height=720)

#Displays image in a window with a frame of 1280 width and 720 height named output
def display_image(img):
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output', 1280, 720)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Makes the source image have same dimensions as destination image
def get_same_shape_image(src, dst):
    width, height = dst.shape[:2]
    src = image_resize(src, width, height)
    return cv2.resize(src, (height, width))

#Returns a Green Screen mask of a high enough range using HSV color values
def get_green_screen_mask_HSV(img):
    return cv2.inRange(img, np.array([50, 100, 80]), np.array([90, 255, 255]))

#Displays a four quadrant image with the 1st quadrant being the original image
#2nd, 3rd and 4th quadrant correspond to each of the Color channels in the color image
def task1(color_space, image):
    
    #Read the image from current directory
    img = cv2.imread(image)
    
    #Make a copy of the image
    img_copy = np.copy(img)
    
    #Get color space conversion code
    code = get_code(color_space)
    
    #Store each color channel arrays into individual variables
    c1, c2, c3 = get_converted_color_components(img_copy, code)
    
    #Stitch the orginal image and corresponding color channels
    stitched_image = get_stitched_images(img_copy, c1, c2, c3)
    
    #Display the final output in a single viewing window
    display_image(stitched_image)

#Displays a four quadrant image where 1st quadrant -> Green screen image, 2nd quadrant -> Extracted persona with white background
#3rd quadrant -> Scenic image , 4th quadrant -> Chroma Keyed image with a persona on the foreground and the scenic image as background
def task2(green_screen, scenic_image):
    
    #Read green screen and scenic images
    green_screen_img = cv2.imread(green_screen)
    scenic_img = cv2.imread(scenic_image)
    
    #Make the green screen image have same dimension as scenic image
    green_screen_img = get_same_shape_image(green_screen_img, scenic_img)
    
    #Convert the two images to HSV color mapping
    img = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(scenic_img, cv2.COLOR_BGR2HSV)
    
    #Generate a green screen mask
    mask = get_green_screen_mask_HSV(img)
    
    #Make green screen background white while preserving the persona
    img[mask!=0] = [0,0,255]
    
    #Make pixels corresponding the persona black to blend the persona 
    img2[mask==0] = [0,0,0]
    
    #Convert the two images back to BGR color space for proper viewing
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    
    #Display the final output in a single viewing window
    display_image(get_stitched_images(green_screen_img, img, scenic_img, img+img2))

if __name__ == '__main__':
    args = sys.argv[1:]