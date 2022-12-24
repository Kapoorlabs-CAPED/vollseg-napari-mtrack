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
        max_x, max_y = 0, 0
        acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

        images = []
        names = []
        for fname in os.listdir(path):
            if any(fname.endswith(f) for f in acceptable_formats):
                image = imread(os.path.join(path, fname))
                if len(image.shape) == 3:
                    image = image[0]
                max_x = max(max_x, image.shape[1])
                max_y = max(max_y, image.shape[0])
            else:
                print(
                    f"ignoring the file {fname} as it is not a valid image file"
                )

        for fname in os.listdir(path):
            if any(fname.endswith(f) for f in acceptable_formats):
                image = imread(os.path.join(path, fname))
                if len(image.shape) == 3:
                    image = image[0]

                image = np.pad(
                    image,
                    (
                        (0, int(max_y) - image.shape[0]),
                        (0, int(max_x) - image.shape[1]),
                    ),
                    mode="constant",
                )
                images.append(image)
                names.append(os.path.splitext(os.path.basename(fname))[0])
        images_array = np.array(images)

    elif os.path.isfile(path):
        if any(fname.endswith(f) for f in acceptable_formats):
            images_array = imread(path)
            if len(images_array.shape) == 3:
                images_array = images_array[0]
        else:
            print(f"ignoring the file {fname} as it is not a valid image file")

    add_kwargs = {}

    layer_type = "image"

    return [(images_array, add_kwargs, layer_type)]
