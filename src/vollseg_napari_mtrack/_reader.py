"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers.
"""
import os

import numpy as np
from tifffile import imread


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str

    Returns
    -------
    function
    """

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):

    if os.path.isdir(path):

        # Set the maximum dimensions to the minimum possible values
        max_x, max_y = float("inf"), float("inf")
        acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

        images = []
        for file in os.listdir(path):

            if any(file.endswith(f) for f in acceptable_formats):
                image = imread(os.oath.join(path, file))
                max_x = max(max_x, image.shape[1])
                max_y = max(max_y, image.shape[0])
            else:
                print(
                    f"ignoring the file {file} as it is not a valid image file"
                )

        for file in os.listdir(path):
            if any(file.endswith(f) for f in acceptable_formats):
                image = imread(os.path.join(path, file))
                image = image.resize(image, (max_y, max_x))
                images.append(image)

        images_array = np.array(images)

    elif os.path.isfile(path):
        if any(file.endswith(f) for f in acceptable_formats):
            images_array = imread(path)
        else:
            print(f"ignoring the file {file} as it is not a valid image file")

    return images_array
