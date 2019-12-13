from skimage.color import rgb2gray, rgb2lab, lab2rgb
import matplotlib.image as mpimg
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_image_colorspaces(filepath):
    """
    Get RGB, Grayscale, and LAB color spaces from image
    """

    # Read RGB
    rgb = mpimg.imread(filepath).astype(np.float64)/255

    # Get LAB
    lab = rgb2lab(rgb)

    # Get Grayscale
    grayscale = rgb2gray(rgb)

    # Return dictionary of colorspaces
    return rgb, lab, grayscale


def flatten_image(image):
    """
    Flattens image into list of points.
    """

    # Check for grayscale
    if len(image.shape) == 2:
        return image.reshape(image.shape[0] * image.shape[1], 1)

    # Otherwise flatten with all channels
    else:
        return image.reshape(image.shape[0] * image.shape[1], image.shape[2])


def prepare_training_data(input_images, output_images):
    """
    Takes X and Y images and transforms them to model format
    """

    # Make sure input and output data is the same length
    assert len(input_images) == len(output_images)

    X_train = np.array([input_image.flatten().reshape(1,
                                                      input_image.shape[0]*input_image.shape[1]) for input_image in input_images])

    Y_train = np.array([[','.join(str(val) for val in pixel) for pixel in output_image.reshape(
        output_image.shape[0]*output_image.shape[1], 3)] for output_image in output_images])

    return X_train, Y_train
