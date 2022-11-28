import numpy as np

from vollseg_napari_mtrack import plugin_wrapper_mtrack


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    plugin_wrapper_mtrack(viewer)

    assert True
