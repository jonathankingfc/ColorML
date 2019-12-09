from layers.convolutional import Conv2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
import numpy as np

img = mpimg.imread('data/test/berry/5_256.jpg')

lab_image = color.rgb2lab(img)

X = lab_image[:, :, 0]
a = lab_image[:, :, 1]
b = lab_image[:, :, 2]


convLayer = Conv2D(num_filters=2, num_input_channels=3)

predictions = convLayer.forward_pass(lab_image)

a_predicted, b_predicted = predictions[:, :, 0], predictions[:, :, 1]


# # GET ACCURACY
mse = (np.square(a_predicted - a).mean() + np.square(b_predicted - b).mean())/2
print(mse)

predicted_lab = np.stack((X, a_predicted, b_predicted), axis=-1)

predicted_rgb = color.lab2rgb(predicted_lab)

plt.imshow(predicted_rgb)

plt.show()
# print(convLayer.forward_pass(lab_image).shape)

# DO MODEL STUFF

# GET ACCURACY

# a_predicted = model_output[:, :, 1]

# error =
# b_predicted = model_output[:, :, 2]
