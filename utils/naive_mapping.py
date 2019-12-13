import numpy as np


def apply_naive_mapping(kmeans_1, kmeans_2, image):

    mapping = dict(zip(np.array(kmeans_1.get_centers()).flatten(),
                       tuple(kmeans_2.get_centers())))

    mapped_image = np.zeros(
        (image.shape[0], image.shape[1], 3))

    for i in range(mapped_image.shape[0]):
        for j in range(mapped_image.shape[1]):
            mapped_image[i, j] = mapping[image[i, j]]

    return mapped_image
