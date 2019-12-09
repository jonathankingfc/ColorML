import numpy as np


class Conv2D:
    """
    This class represents a layer that will perform 2-D convolutions in the output of the previous layer
    """

    def __init__(self, num_filters, num_input_channels):

        self.num_filters = num_filters

        self.filters = np.random.randn(
            num_filters, 3, 3, num_input_channels) / 9*num_input_channels

        self.last_input = None

    def generate_conv_regions(self, input_volume):
        """
        Generates all 3x3xC convolutions from the input volume where C is the number of channels in the input volume.
        """
        h, w, _ = input_volume.shape

        padded_input = np.pad(input_volume, ((2, 2), (2, 2), (0, 0)),
                              mode='constant', constant_values=0)

        for i in range(h):
            for j in range(w):
                conv_region = padded_input[i:(i + 3), j:(j + 3), :]
                yield conv_region, i, j

    def forward_pass(self, input_volume):
        """
        Performs a forward pass of the layer on the input volume.
        """

        self.last_input = input_volume

        h, w, _ = input_volume.shape
        output_volume = np.zeros((h, w, self.num_filters))

        for conv_region, i, j in self.generate_conv_regions(input_volume):
            output_volume[i, j] = np.sum(
                conv_region * self.filters, axis=(1, 2, 3))

        return output_volume
