from vollseg_napari_mtrack import plugin_wrapper_mtrack


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_plugin_wrapper_mtrack(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    # create our widget, passing in the viewer
    my_widget = plugin_wrapper_mtrack()
    viewer.window.add_dock_widget(my_widget)
