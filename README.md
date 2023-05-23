Author: Jonah Biedermann, Stuart Allen, and John Kaufman

1. Make sure to install all requirements such as openCV,  numpy, and os. 
2. make sure you are using python3 and run ```python3 BB_crop.py```
3. Then just follow the instructions given by the program. Assume all paths specified are in the same directory.



augmentation.py:

A data augmentation program, used to create copies of images from real images for the purpose of expanding the training and validation datasets. R. In the case of the weld image dataset, the augmentation process consists of making a copy of a pre-existing image, and putting it through a series of transformations to create a new image.

Dependencies: scikit-image, PIL, random

The program takes images from the source directory, and applies a series of pseudo-randomly selected transformations to create new copies of the images and store them in the target directory.

The augmentations in this file are:
    Vertical Flip (Flip an image on the y-axis)
    Horizontal Flip (Flip an image on the x-axis)
    Convert to Gray (Convert an image to black and white)
    Crop (Crop an image)
    Rotate (Rotate an image)
    Blur (Blur an image)


There are several global control variables to help control the program.
    VERBOSE_OUTPUT controls whether to print to the terminal each transformation and information about the files.
    ALL_TO_GRAY can be used to control that all images are converted to gray, no matter what.
    ROTATION_MAX defines the maximum number of degrees an image can be rotated in either direction.
    POSSIBLE_OPERATIONS is a numeric list representing the operations to be executed. Add new operations to this list.
    SOURCE_IMAGE_DIRECTORY is the path to the directory where the source images are located, relative to augmentation.py
    AUGMENTED_IMAGE_DIRECTORY is the path to the directory where the augmented images should be stored.



GAN_augmentations.ipynb

This program is an implementation of a Generative Adversarial Network (GAN), a data augmentation machine learning program to create new images from reference images. The model works by training two neural networks, one with the goal of creating images from reference images that cannot be differentiated from real images (called the generator), and one that learns to differentiate real images from those created by a network (called the discriminator). These two networks are trained simultaneously, to create images indistinguishable from real images.

This code is adapted from the guide at this address, which gives a more in-depth explanation of how the code works and what is happening with the GAN: https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b

To run this code, download the ipynb file, and upload it to google colab, which will have all of the required dependencies pre-installed. When using google colab, ensure it is running the code on the device GPU, to help quicken the training process. The only other requirement is that the images need to be a 28x28 pixel resolution.

