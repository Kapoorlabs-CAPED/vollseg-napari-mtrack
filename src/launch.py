import napari


def show_napari():
    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget("vollseg-napari-mtrack", "MTrack")
    napari.run()


if __name__ == "__main__":

    show_napari()
