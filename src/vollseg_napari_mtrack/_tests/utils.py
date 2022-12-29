import numpy as np


def line_image(shape=(128, 128), slope=None, intercept=None):
    if slope is None:
        slope = [1, 3, -4]
    if intercept is None:
        intercept = [0, 6, 4]
    if type(shape) is list:
        shape = tuple(shape)
    xs = np.arange(shape[0])
    img = np.zeros(shape=(shape[0], shape[1]))
    for i in range(len(xs)):
        x = xs[i]
        for m in slope:
            for c in intercept:
                y = m * x + c
                if y > 0 and x > 0 and y < shape[0] and x < shape[1]:
                    img[int(y), int(x)] = 1

    return img
