import napari


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget():
    # make viewer and add an image layer using our fixture
    viewer = napari.Viewer()
    # create our widget, passing in the viewer
    viewer.window.add_plugin_dock_widget("vollseg-napari-mtrack", "MTrack")
    napari.run()


if __name__ == "__main__":
    test_example_q_widget()
