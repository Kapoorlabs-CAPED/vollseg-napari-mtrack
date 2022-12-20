"""
VollSeg Napari MTrack .

Made by Kapoorlabs, 2022
"""

import functools
import time
from pathlib import Path
from typing import List, Union

import napari
import numpy as np
from magicgui import magicgui
from magicgui import widgets as mw
from napari.qt.threading import thread_worker
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy


def plugin_wrapper_mtrack():

    from caped_ai_mtrack.RansacModels import LinearFunction, QuadraticFunction
    from csbdeep.utils import load_json
    from vollseg import UNET
    from vollseg.pretrained import get_model_folder, get_registered_models

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

    _models_vollseg, _aliases_vollseg = get_registered_models(UNET)

    models_vollseg = [
        ((_aliases_vollseg[m][0] if len(_aliases_vollseg[m]) > 0 else m), m)
        for m in _models_vollseg
    ]

    worker = None
    model_vollseg_configs = dict()
    model_selected_vollseg = None
    # model_selected_ransac = None

    PRETRAINED = UNET
    CUSTOM_VOLLSEG = "CUSTOM_VOLLSEG"
    vollseg_model_type_choices = [
        ("PreTrained", PRETRAINED),
        ("Custom U-Net", CUSTOM_VOLLSEG),
        ("None", "NOSEG"),
    ]

    ransac_model_type_choices = [
        ("Linear", LinearFunction),
        ("Quadratic", QuadraticFunction),
    ]

    DEFAULTS_MODEL = dict(
        vollseg_model_type=UNET,
        model_vollseg=models_vollseg[0][0],
        model_vollseg_none="NOSEG",
        axes="YX",
        ransac_model_type=LinearFunction,
    )

    DEFAULTS_PRED_PARAMETERS = dict(
        norm_image=True,
        n_tiles=(1, 1),
        max_error=2,
        min_num_time_points=20,
        maximum_gap=4,
    )

    @functools.lru_cache(maxsize=None)
    def get_model_vollseg(vollseg_model_type, model_vollseg):
        if vollseg_model_type == CUSTOM_VOLLSEG:
            path_vollseg = Path(model_vollseg)
            path_vollseg.is_dir() or _raise(
                FileNotFoundError(f"{path_vollseg} is not a directory")
            )

            model_class_vollseg = UNET
            return model_class_vollseg(
                None, name=path_vollseg.name, basedir=str(path_vollseg.parent)
            )

        elif vollseg_model_type != DEFAULTS_MODEL["model_vollseg_none"]:
            return vollseg_model_type.local_from_pretrained(model_vollseg)
        else:
            return None

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.1038/s41598-018-37767-1")

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h1><a href=" {citation}"> MTrack: Automated Detection, Tracking, and Analysis of Dynamic Microtubules</a></h1>',
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
        model_vollseg=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained UNET Model",
            choices=models_vollseg,
            value=DEFAULTS_MODEL["model_vollseg"],
        ),
        model_vollseg_none=dict(
            widget_type="Label", visible=False, label="NOSEG"
        ),
        model_folder=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom VollSeg",
            mode="d",
        ),
        ransac_model_type=dict(
            widget_type="RadioButtons",
            label="Ransac Model Type",
            orientation="horizontal",
            choices=ransac_model_type_choices,
            value=DEFAULTS_MODEL["ransac_model_type"],
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
        model_vollseg,
        model_vollseg_none,
        model_folder,
        ransac_model_type,
        ransac_model_linear,
        n_tiles,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:
        # x = get_data(image)

        nonlocal worker

        plugin.label_head.native.setOpenExternalLinks(True)
        plugin.label_head.native.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )

        widget_for_vollseg_modeltype = {
            UNET: plugin.model_vollseg,
            "NOSEG": plugin.model_vollseg_none,
            CUSTOM_VOLLSEG: plugin.model_folder,
        }

        def select_model_vollseg(key):
            nonlocal model_selected_vollseg
            if key is not None:
                model_selected_vollseg = key
                # config_vollseg = model_vollseg_configs.get(key)
                # update_vollseg(
                #   "model_vollseg", config_vollseg is not None, config_vollseg
                # )
            if (
                plugin.vollseg_model_type.value
                == DEFAULTS_MODEL["model_vollseg_none"]
            ):
                model_selected_vollseg = None

        def widgets_inactive(*widgets, active):
            for widget in widgets:
                widget.visible = active

        def widgets_valid(*widgets, valid):
            for widget in widgets:
                widget.native.setStyleSheet(
                    "" if valid else "background-color: red"
                )

        @change_handler(plugin.vollseg_model_type, init=False)
        def _seg_model_type_change(seg_model_type: Union[str, type]):
            selected = widget_for_vollseg_modeltype[seg_model_type]
            for w in {
                plugin.model_vollseg,
                plugin.model_vollseg_none,
                plugin.model_folder,
            } - {selected}:
                w.hide()

            selected.show()

            # Trigger model change
            selected.changed(selected.value)

        @change_handler(plugin.model_vollseg, plugin.model_vollseg_none)
        def _seg_model_change(model_name: str):

            if Signal.sender() is not plugin.model_vollseg_none:

                model_class_vollseg = UNET
                key = model_class_vollseg, model_name

                if key not in model_vollseg_configs:

                    @thread_worker
                    def _get_model_folder():
                        return get_model_folder(*key)

                    def _process_model_folder(path):

                        try:
                            model_vollseg_configs[key] = load_json(
                                str(path / "config.json")
                            )
                        finally:
                            select_model_vollseg[key]
                            plugin.progress_bar.hide()

                    worker = _get_model_folder()
                    worker.returned.connect(_process_model_folder)
                    worker.start()

                    # delay showing progress bar -> won't show up if model already downloaded
                    # TODO: hacky -> better way to do this?
                    time.sleep(0.1)
                    plugin.call_button.enabled = False
                    plugin.progress_bar.label = "Downloading UNET model"
                    plugin.progress_bar.show()

                else:
                    select_model_vollseg(key)

            else:
                select_model_vollseg(None)
                plugin.call_button.enabled = True
                plugin.model_folder_vollseg.line_edit.tooltip = (
                    "Invalid model directory"
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
