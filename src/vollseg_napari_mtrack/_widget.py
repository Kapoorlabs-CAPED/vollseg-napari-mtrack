"""
VollSeg Napari MTrack .

Made by Kapoorlabs, 2022
"""

import functools
from pathlib import Path
from typing import List

import napari
import numpy as np
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy


def plugin_wrapper_mtrack():

    from vollseg import UNET
    from vollseg.pretrained import get_registered_models

    DEBUG = True

    def _raise(e):
        if isinstance(e, BaseException):
            raise e
        else:
            raise ValueError(e)

    def get_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("image loaded")
        return np.asarray(image)

    def abspath(root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:
                    print(f"{str(emitter.name).upper()}: {source.name}")
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    _models_vollnet, _aliases_vollnet = get_registered_models(UNET)

    models_vollnet = [
        ((_aliases_vollnet[m][0] if len(_aliases_vollnet[m]) > 0 else m), m)
        for m in _models_vollnet
    ]

    worker = None
    PRETRAINED = "PRETRAINED"
    CUSTOM_VOLLSEG = "CUSTOM_VOLLSEG"

    DEFAULTS_MODEL = dict(
        vollseg_model_class=UNET,
        oneat_model_type=CUSTOM_VOLLSEG,
        model_vollnet=models_vollnet[0][0],
        axes="YX",
    )

    vollseg_model_type_choices = [
        ("PreTrained", PRETRAINED),
        ("Custom U-Net", CUSTOM_VOLLSEG),
    ]

    DEFAULTS_PRED_PARAMETERS = dict(
        norm_image=True,
        n_tiles=(1, 1),
        max_error=2,
        min_num_time_points=20,
        maximum_gap=4,
    )

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.1242/focalplane.10759")

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h5><a href=" {citation}"> FocalPlane</a></h5>',
        ),
        image=dict(label="Input Image"),
        max_error=dict(
            widget_type="FloatSpinBox",
            label="Max error",
            min=0.0,
            step=5,
            value=DEFAULTS_PRED_PARAMETERS["max_error"],
        ),
        min_num_time_points=dict(
            widget_type="FloatSpinBox",
            label="Minimum number of timepoints",
            min=0.0,
            step=5,
            value=DEFAULTS_PRED_PARAMETERS["min_num_time_points"],
        ),
        maximum_gap=dict(
            widget_type="SpinBox",
            label="Maximum gap b/w events",
            min=0,
            step=1,
            value=DEFAULTS_PRED_PARAMETERS["maximum_gap"],
        ),
        vollseg_model_type=dict(
            widget_type="RadioButtons",
            label="VollSeg Model Type",
            orientation="horizontal",
            choices=vollseg_model_type_choices,
            value=DEFAULTS_MODEL["vollseg_model_type"],
        ),
        model_folder=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom VollSeg",
            mode="d",
        ),
        n_tiles=dict(
            widget_type="LiteralEvalLineEdit",
            label="Number of Tiles",
            value=DEFAULTS_PRED_PARAMETERS["n_tiles"],
        ),
        defaults_model_button=dict(
            widget_type="PushButton", text="Restore Model Defaults"
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=True,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        max_error,
        min_num_time_points,
        maximum_gap,
        vollseg_model_type,
        model_folder,
        n_tiles,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:
        # x = get_data(image)

        nonlocal worker

        """
            widget_for_modeltype = {
                NEATVollNet: plugin.model_vollnet,
                NEATLRNet: plugin.model_lrnet,
                NEATTResNet: plugin.model_tresnet,
                NEATResNet: plugin.model_resnet,
                CSV_PREDICTIONS: plugin.csv_folder,
                CUSTOM_NEAT: plugin.model_folder,
            }
            """

        plugin.label_head.native.setOpenExternalLinks(True)
        plugin.label_head.native.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )

        def widgets_inactive(*widgets, active):
            for widget in widgets:
                widget.visible = active

        def widgets_valid(*widgets, valid):
            for widget in widgets:
                widget.native.setStyleSheet(
                    "" if valid else "background-color: red"
                )

        @change_handler(plugin.max_error)
        def _max_error_change(value: float):
            plugin.max_error.value = value

        @change_handler(plugin.min_num_time_points)
        def _min_num_time_points(value: float):
            plugin.min_num_time_points.value = value

        @change_handler(plugin.maximum_gap)
        def _maximum_gap(value: float):
            plugin.maximum_gap.value = value

        def restore_prediction_parameters_defaults():
            for k, v in DEFAULTS_PRED_PARAMETERS.items():
                getattr(plugin, k).value = v

        @change_handler(plugin.image, init=False)
        def _image_change(image: napari.layers.Image):
            plugin.image.tooltip = (
                f"Shape: {get_data(image).shape, str(image.name)}"
            )

    return plugin
