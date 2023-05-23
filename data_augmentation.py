import os
import cv2
from skimage.color import rgb2lab, lab2rgb
from PIL import Image, ImageFilter, ImageOps
import random

VERBOSE_OUTPUT = False #Whether to output information about augmentations
ALL_TO_GRAY = False #whether to convert all data points to gray
ROTATION_MAX = 30 #the maximum number of degrees to rotate an image in one direction or another
POSSIBLE_OPERATIONS = [1, 2, 3, 4, 5, 6]
SOURCE_IMAGE_DIRECTORY = './images/' #Location of the source images
AUGMENTED_IMAGE_DIRECTORY = './augmented_images/' #Location to store the augmented images

#For each file in the source directory
for file in os.listdir(SOURCE_IMAGE_DIRECTORY):
    augmented_img = Image.open(SOURCE_IMAGE_DIRECTORY + file)


    #Choose a random number between 1 and 6, we will perform that many operations
    num_operations = random.choice(POSSIBLE_OPERATIONS)

    #Sample a number of operations equal to num_operations from POSSIBLE_OPERATIONS. These will be the augmentations performed on the image
    #If 1 is selected, then the image will be flipped vertically, etc.
    operations = (random.sample(POSSIBLE_OPERATIONS, num_operations))

    if VERBOSE_OUTPUT:
        print("\n=============================================\n")
        print(file)
        print(str(num_operations) + " data augmentations")
        print(operations)

    #Flip Vertically
    if 1 in operations:
        augmented_img = ImageOps.flip(augmented_img)

        if VERBOSE_OUTPUT:
            print("Flipping Vertically")

    #Flip Horizontally
    if 2 in operations:
        augmented_img = ImageOps.mirror(augmented_img)

        if VERBOSE_OUTPUT:
            print("Flipping Horizontally")

    #Convert to Gray
    if (3 in operations) or (ALL_TO_GRAY):
        augmented_img = augmented_img.convert('L')

        if VERBOSE_OUTPUT:
            print("Converting to Gray scale")

    #Crop
    if 4 in operations:
        width, height = augmented_img.size
        # Setting the points for cropped image
        left = width*0.20
        top = height*0.25
        right = width*0.75
        bottom = height*0.70
        augmented_img = augmented_img.crop((left, top, right, bottom))

        if VERBOSE_OUTPUT:
            print("Cropping")

    #Rotate
    if 5 in operations:
        rotate_amount =  random.randint((-1*ROTATION_MAX), ROTATION_MAX)
        augmented_img = augmented_img.rotate(rotate_amount)

        if VERBOSE_OUTPUT:
            print("Rotating " + str(rotate_amount) + " degrees")

    #Blur
    if 6 in operations:
        augmented_img = augmented_img.filter(ImageFilter.BLUR)

        if VERBOSE_OUTPUT:
            print("Blurring")

    augmented_img.save(AUGMENTED_IMAGE_DIRECTORY + file + '_augmented' + '.jpg')
