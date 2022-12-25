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
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget


def plugin_wrapper_mtrack():

    from caped_ai_mtrack.Fits import Ransac
    from caped_ai_mtrack.RansacModels import LinearFunction, QuadraticFunction
    from csbdeep.utils import axes_check_and_normalize, axes_dict, load_json
    from skimage.morphology import thin
    from vollseg import UNET, VollSeg
    from vollseg.pretrained import get_model_folder, get_registered_models

    DEBUG = False

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

    PRETRAINED = UNET
    CUSTOM_VOLLSEG = "CUSTOM_VOLLSEG"
    vollseg_model_type_choices = [
        ("PreTrained", PRETRAINED),
        ("Custom U-Net", CUSTOM_VOLLSEG),
        ("NOSEG", "NOSEG"),
    ]

    ransac_model_type_choices = [
        ("Linear", LinearFunction),
        ("Quadratic", QuadraticFunction),
    ]

    DEFAULTS_MODEL = dict(
        vollseg_model_type=CUSTOM_VOLLSEG,
        model_vollseg=models_vollseg[0][0],
        model_vollseg_none="NOSEG",
        axes="TYX",
        ransac_model_type=LinearFunction,
    )

    model_selected_ransac = DEFAULTS_MODEL["ransac_model_type"]
    DEFAULTS_SEG_PARAMETERS = dict(n_tiles=(1, 1, 1))

    DEFAULTS_PRED_PARAMETERS = dict(
        max_error=0.1,
        min_num_time_points=2,
        minimum_height=4,
        time_axis=1,
    )

    def get_model_ransac(ransac_model_type):

        return ransac_model_type

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

    @magicgui()
    def plugin_table():
        return plugin_table

    @magicgui()
    def plugin_plots():
        return plugin_plots

    @magicgui(
        max_error=dict(
            widget_type="FloatSpinBox",
            label="Max error",
            min=0.0,
            step=0.05,
            value=DEFAULTS_PRED_PARAMETERS["max_error"],
        ),
        min_num_time_points=dict(
            widget_type="SpinBox",
            label="Minimum number of timepoints",
            min=0.0,
            step=5,
            value=DEFAULTS_PRED_PARAMETERS["min_num_time_points"],
        ),
        minimum_height=dict(
            widget_type="SpinBox",
            label="Minimum height for catastrophe event",
            min=0,
            step=1,
            value=DEFAULTS_PRED_PARAMETERS["minimum_height"],
        ),
        time_axis=dict(
            widget_type="SpinBox",
            label="Kymograph time axis (0 for along y, 1 for along x)",
            min=0,
            step=1,
            value=DEFAULTS_PRED_PARAMETERS["time_axis"],
        ),
        ransac_model_type=dict(
            widget_type="RadioButtons",
            label="Ransac Model Type",
            orientation="horizontal",
            choices=ransac_model_type_choices,
            value=DEFAULTS_MODEL["ransac_model_type"],
        ),
        defaults_params_button=dict(
            widget_type="PushButton", text="Restore Parameter Defaults"
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def plugin_ransac_parameters(
        max_error,
        min_num_time_points,
        minimum_height,
        time_axis,
        ransac_model_type,
        defaults_params_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        return plugin_ransac_parameters

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.1038/s41598-018-37767-1")

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h5><a href=" {citation}"> MTrack: Automated Detection, Tracking, and Analysis of Dynamic Microtubules</a></h5>',
        ),
        image=dict(label="Input Image"),
        axes=dict(
            widget_type="LineEdit",
            label="Image Axes",
            value=DEFAULTS_MODEL["axes"],
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
        model_folder_vollseg=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom VollSeg",
            mode="d",
        ),
        n_tiles=dict(
            widget_type="LiteralEvalLineEdit",
            label="Number of Tiles",
            value=DEFAULTS_SEG_PARAMETERS["n_tiles"],
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
        axes,
        vollseg_model_type,
        model_vollseg,
        model_vollseg_none,
        model_folder_vollseg,
        n_tiles,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        x = get_data(image)
        print(x.shape)
        axes = axes_check_and_normalize(axes, length=x.ndim)
        nonlocal worker
        progress_bar.label = "Starting MTrack"
        if model_selected_vollseg is not None:
            vollseg_model = get_model_vollseg(*model_selected_vollseg)

        axes_out = None
        if vollseg_model is not None:
            assert vollseg_model._axes_out[-1] == "C"
            axes_out = list(vollseg_model._axes_out[:-1])
        scale_in_dict = dict(zip(axes, image.scale))
        scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]
        ransac_model = get_model_ransac(model_selected_ransac)

        if "T" in axes:
            t = axes_dict(axes)["T"]
            n_frames = x.shape[t]

            def progress_thread(current_time):

                progress_bar.label = "Fitting Functions (files)"
                progress_bar.range = (0, n_frames)
                progress_bar.value = current_time + 1
                progress_bar.show()

            def progress_thread_ransac(current_time):

                progress_bar.label = "Fitting Function Fits (files)"
                progress_bar.range = (0, n_frames)
                progress_bar.value = current_time + 1
                progress_bar.show()

        if "T" in axes and axes_out is not None:
            x_reorder = np.moveaxis(x, t, 0)

            axes_reorder = axes.replace("T", "")
            axes_out.insert(t, "T")
            # determine scale for output axes
            scale_in_dict = dict(zip(axes, image.scale))
            scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]
            worker = _Unet_time(
                vollseg_model,
                x_reorder,
                axes_reorder,
                scale_out,
                t,
                x,
                ransac_model,
            )
            worker.returned.connect(return_segment_unet_time)
            worker.yielded.connect(progress_thread)

        else:
            worker = _Unet(vollseg_model, x, axes, scale_out, ransac_model)
            worker.returned.connect(return_segment_unet)

        progress_bar.hide()

    plugin.label_head.value = '<br>Citation <tt><a href="https://doi.org/10.1038/s41598-018-37767-1" style="color:gray;">MTrack Nature</a></tt>'
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )

    def return_segment_unet_time(pred):

        layer_data, time_line_locations, scale_out = pred

        name_remove = ["Fits_MTrack", "Seg_MTrack"]
        for layer in list(plugin.viewer.value.layers):
            if any(name in layer.name for name in name_remove):
                plugin.viewer.value.layers.remove(layer)

        plugin.viewer.value.add_labels(layer_data, name="Seg_MTrack")

        plugin.viewer.value.add_shapes(
            np.asarray(time_line_locations),
            name="Fits_MTrack",
            shape_type="line",
            face_color=[0] * 4,
            edge_color="red",
            edge_width=1,
        )

    def return_segment_unet(pred):

        layer_data, line_locations, scale_out = pred

        name_remove = ["Fits_MTrack", "Seg_MTrack"]
        for layer in list(plugin.viewer.value.layers):
            if any(name in layer.name for name in name_remove):
                plugin.viewer.value.layers.remove(layer)

        plugin.viewer.value.add_labels(layer_data, name="Seg_MTrack")

        plugin.viewer.value.add_shapes(
            np.asarray(line_locations),
            name="Fits_MTrack",
            shape_type="line",
            face_color=[0] * 4,
            edge_color="red",
            edge_width=1,
        )

    @thread_worker(connect={"returned": return_segment_unet_time})
    def _Unet_time(
        model_unet, x_reorder, axes_reorder, scale_out, t, x, ransac_model
    ):
        pre_res = []
        yield 0

        correct_label_present = []
        any_label_present = []
        for layer in list(plugin.viewer.value.layers):
            if (
                isinstance(layer, napari.layers.Labels)
                and layer.data.shape == get_data(plugin.image.value).shape
            ):
                correct_label_present.append(True)
            elif (
                isinstance(layer, napari.layers.Labels)
                and layer.data.shape != get_data(plugin.image.value).shape
            ):
                correct_label_present.append(False)

            if not isinstance(layer, napari.layers.Labels):
                any_label_present.append(False)
            elif isinstance(layer, napari.layers.Labels):
                any_label_present.append(True)
        print(correct_label_present, any_label_present)
        if (
            any(correct_label_present) is False
            or any(any_label_present) is False
        ):

            for count, _x in enumerate(x_reorder):

                pre_res.append(
                    VollSeg(
                        _x,
                        unet_model=model_unet,
                        n_tiles=plugin.n_tiles.value,
                        axes=axes_reorder,
                    )
                )

            unet_mask, skeleton, denoised_image = zip(*pre_res)

            unet_mask = np.asarray(unet_mask)

            unet_mask = unet_mask > 0
            unet_mask = np.moveaxis(unet_mask, 0, t)
            unet_mask = np.reshape(unet_mask, x.shape)
            for i in range(unet_mask.shape[0]):
                unet_mask[i] = thin(unet_mask[i])

            skeleton = np.asarray(skeleton)
            skeleton = skeleton > 0
            skeleton = np.moveaxis(skeleton, 0, t)
            skeleton = np.reshape(skeleton, x.shape)

            denoised_image = np.asarray(denoised_image)
            denoised_image = np.moveaxis(denoised_image, 0, t)
            denoised_image = np.reshape(denoised_image, x.shape)

            layer_data = unet_mask

        else:
            for layer in list(plugin.viewer.value.layers):
                if (
                    isinstance(layer, napari.layers.Labels)
                    and layer.data.shape == get_data(plugin.image.value).shape
                ):

                    layer_data = layer.data

        print("Model", ransac_model)
        if ransac_model == LinearFunction:
            degree = 2
        if ransac_model == QuadraticFunction:
            degree = 3

        time_estimators = {}
        time_estimator_inliers = {}
        time_line_locations = []
        for count, i in enumerate(range(layer_data.shape[0])):
            yield count
            non_zero_indices = list(zip(*np.where(layer_data[i] > 0)))
            sorted_non_zero_indices = sorted(
                non_zero_indices,
                key=lambda x: x[plugin_ransac_parameters.time_axis.value],
            )
            if len(sorted_non_zero_indices) > 0:
                yarray, xarray = zip(*sorted_non_zero_indices)

                ransac_result = Ransac(
                    sorted_non_zero_indices,
                    ransac_model,
                    degree,
                    min_samples=plugin_ransac_parameters.min_num_time_points.value,
                    max_trials=100,
                    iterations=10,
                    residual_threshold=plugin_ransac_parameters.max_error.value,
                    save_name="",
                )
                (
                    estimators,
                    estimator_inliers,
                ) = ransac_result.extract_multiple_lines()
                time_estimators[i] = estimators
                time_estimator_inliers[i] = estimator_inliers

                line_locations = []
                for j in range(len(estimators)):

                    estimator = estimators[j]
                    estimator_inlier = estimator_inliers[j]
                    estimator_inliers_list = np.copy(estimator_inlier)
                    yarray, xarray = zip(*estimator_inliers_list.tolist())
                    yarray = np.asarray(yarray)
                    xarray = np.asarray(xarray)
                    line_locations.append(
                        [
                            [estimator.predict(xarray[0]), xarray[0]],
                            [estimator.predict(xarray[-1]), xarray[-1]],
                        ]
                    )
                    time_line_locations.append(
                        [
                            [i, estimator.predict(xarray[0]), xarray[0]],
                            [i, estimator.predict(xarray[-1]), xarray[-1]],
                        ]
                    )

        pred = layer_data, time_line_locations, scale_out
        return pred

    @thread_worker(connect={"returned": return_segment_unet})
    def _Unet(model_unet, x, axes, scale_out, ransac_model):

        correct_label_present = []
        any_label_present = []
        for layer in list(plugin.viewer.value.layers):
            if (
                isinstance(layer, napari.layers.Labels)
                and layer.data.shape == get_data(plugin.image.value).shape
            ):
                correct_label_present.append(True)
            elif (
                isinstance(layer, napari.layers.Labels)
                and layer.data.shape != get_data(plugin.image.value).shape
            ):
                correct_label_present.append(False)

            if not isinstance(layer, napari.layers.Labels):
                any_label_present.append(False)
            elif isinstance(layer, napari.layers.Labels):
                any_label_present.append(True)
        if (
            any(correct_label_present) is False
            or any(any_label_present) is False
        ):

            res = VollSeg(
                x,
                unet_model=model_unet,
                n_tiles=plugin.n_tiles.value,
                axes=axes,
            )

            unet_mask, skeleton, denoised_image = res

            layer_data = thin(unet_mask)

        else:
            for layer in list(plugin.viewer.value.layers):
                if (
                    isinstance(layer, napari.layers.Labels)
                    and layer.data.shape == get_data(plugin.image.value).shape
                ):

                    layer_data = layer.data

        non_zero_indices = list(zip(*np.where(layer_data > 0)))
        sorted_non_zero_indices = sorted(
            non_zero_indices,
            key=lambda x: x[plugin_ransac_parameters.time_axis.value],
        )
        yarray, xarray = zip(*sorted_non_zero_indices)

        if ransac_model == LinearFunction:
            degree = 2
        if ransac_model == QuadraticFunction:
            degree = 3

        ransac_result = Ransac(
            sorted_non_zero_indices,
            ransac_model,
            degree,
            min_samples=plugin_ransac_parameters.min_num_time_points.value,
            max_trials=100,
            iterations=10,
            residual_threshold=plugin_ransac_parameters.max_error.value,
            save_name="",
        )

        estimators, estimator_inliers = ransac_result.extract_multiple_lines()

        line_locations = []
        for i in range(len(estimators)):

            estimator = estimators[i]
            estimator_inlier = estimator_inliers[i]
            estimator_inliers_list = np.copy(estimator_inlier)
            yarray, xarray = zip(*estimator_inliers_list.tolist())
            yarray = np.asarray(yarray)
            xarray = np.asarray(xarray)
            line_locations.append(
                [
                    [estimator.predict(xarray[0]), xarray[0]],
                    [estimator.predict(xarray[-1]), xarray[-1]],
                ]
            )

        pred = layer_data, line_locations, scale_out
        return pred

    widget_for_vollseg_modeltype = {
        UNET: plugin.model_vollseg,
        "NOSEG": plugin.model_vollseg_none,
        CUSTOM_VOLLSEG: plugin.model_folder_vollseg,
    }

    tabs = QTabWidget()

    parameter_ransac_tab = QWidget()
    _parameter_ransac_tab_layout = QVBoxLayout()
    parameter_ransac_tab.setLayout(_parameter_ransac_tab_layout)
    _parameter_ransac_tab_layout.addWidget(plugin_ransac_parameters.native)
    tabs.addTab(parameter_ransac_tab, "Ransac Parameter Selection")

    plots_tab = QWidget()
    _plots_tab_layout = QVBoxLayout()
    plots_tab.setLayout(_plots_tab_layout)
    _plots_tab_layout.addWidget(plugin_plots.native)
    tabs.addTab(plots_tab, "Ransac Plots")

    table_tab = QWidget()
    _table_tab_layout = QVBoxLayout()
    table_tab.setLayout(_table_tab_layout)
    _table_tab_layout.addWidget(plugin_table.native)
    tabs.addTab(table_tab, "Table")

    plugin.native.layout().addWidget(tabs)

    def select_model_ransac(key):
        nonlocal model_selected_ransac
        model_selected_ransac = key

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    class Updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in ("image_axes", "model_vollseg", "n_tiles")
                }
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f"HELP: {msg}")

        def _update(self):

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print("GOT viewer")

            def _model(valid):
                widgets_valid(
                    plugin.model_vollseg,
                    plugin.model_folder_vollseg.line_edit,
                    valid=valid,
                )
                if valid:
                    config_vollseg = self.args.model_vollseg
                    axes_vollseg = config_vollseg.get(
                        "axes",
                        "YXC"[-len(config_vollseg["unet_input_shape"]) :],
                    )

                    plugin.model_folder_vollseg.line_edit.tooltip = ""
                    return axes_vollseg, config_vollseg
                else:
                    plugin.model_folder_vollseg.line_edit.tooltip = (
                        "Invalid model directory"
                    )

            def _image_axes(valid):
                axes, image, err = getattr(
                    self.args, "image_axes", (None, None, None)
                )

                widgets_valid(
                    plugin.axes,
                    valid=(
                        valid
                        or (image is None and (axes is None or len(axes) == 0))
                    ),
                )

                if valid:
                    plugin.axes.tooltip = "\n".join(
                        [
                            f"{a} = {s}"
                            for a, s in zip(axes, get_data(image).shape)
                        ]
                    )
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith(".") else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ""

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, "n_tiles", (1, 1, 1))
                widgets_valid(plugin.n_tiles, valid=(valid or image is None))
                if valid:
                    plugin.n_tiles.tooltip = "\n".join(
                        [
                            f"{t}: {s}"
                            for t, s in zip(n_tiles, get_data(image).shape)
                        ]
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ""
                    plugin.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _restore():
                widgets_valid(
                    plugin.image, valid=plugin.image.value is not None
                )

            all_valid = False
            help_msg = ""

            if (
                self.valid.image_axes
                and self.valid.n_tiles
                and self.valid.model_vollseg
            ):
                axes_image, image = _image_axes(True)
                (axes_model_vollseg, config_vollseg) = _model(True)
                n_tiles = _n_tiles(True)
                if not _no_tiling_for_axis(axes_image, n_tiles, "C"):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = "number of tiles must be 1 for C axis"
                    plugin.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, "T"):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = "number of tiles must be 1 for T axis"
                    plugin.n_tiles.tooltip = err
                    _restore()

                else:
                    # check if image and models are compatible
                    ch_model_vollseg = config_vollseg["n_channel_in"]

                    ch_image = (
                        get_data(image).shape[axes_dict(axes_image)["C"]]
                        if "C" in axes_image
                        else 1
                    )
                    all_valid = (
                        set(axes_model_vollseg.replace("C", ""))
                        == set(axes_image.replace("C", "").replace("T", ""))
                        and ch_model_vollseg == ch_image
                    )

                    widgets_valid(
                        plugin.image,
                        plugin.model_vollseg,
                        plugin.model_folder_vollseg.line_edit,
                        valid=all_valid,
                    )
                    if all_valid:
                        help_msg = ""
                    else:
                        help_msg = f'Model with axes {axes_model_vollseg.replace("C", f"C[{ch_model_vollseg}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:

                _image_axes(self.valid.image_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model_vollseg)

                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(
                    f"valid ({all_valid}):",
                    ", ".join(
                        [f"{k}={v}" for k, v in vars(self.valid).items()]
                    ),
                )

    update_vollseg = Updater()

    def select_model_vollseg(key):
        nonlocal model_selected_vollseg
        if key is not None:
            model_selected_vollseg = key
            config_vollseg = model_vollseg_configs.get(key)
            update_vollseg(
                "model_vollseg", config_vollseg is not None, config_vollseg
            )
        if (
            plugin.vollseg_model_type.value
            == DEFAULTS_MODEL["model_vollseg_none"]
        ):
            model_selected_vollseg = None

    @change_handler(plugin_ransac_parameters.ransac_model_type, init=False)
    def _ransac_model_change():

        key = plugin_ransac_parameters.ransac_model_type.value
        select_model_ransac(key)

    @change_handler(plugin.vollseg_model_type, init=False)
    def _seg_model_type_change(seg_model_type: Union[str, type]):
        selected = widget_for_vollseg_modeltype[seg_model_type]
        for w in {
            plugin.model_vollseg,
            plugin.model_vollseg_none,
            plugin.model_folder_vollseg,
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
                        select_model_vollseg(key)
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

    @change_handler(plugin.model_folder_vollseg, init=False)
    def _model_vollseg_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_VOLLSEG, path
        try:
            if not path.is_dir():
                return
            model_vollseg_configs[key] = load_json(str(path / "config.json"))
        except FileNotFoundError:
            pass
        finally:
            select_model_vollseg(key)

    @change_handler(plugin_ransac_parameters.max_error)
    def _max_error_change(value: float):
        plugin_ransac_parameters.max_error.value = value

    @change_handler(plugin_ransac_parameters.min_num_time_points)
    def _min_num_time_points(value: int):
        plugin_ransac_parameters.min_num_time_points.value = value

    @change_handler(plugin_ransac_parameters.minimum_height)
    def _minimum_height(value: int):
        plugin_ransac_parameters.minimum_height.value = value

    @change_handler(plugin_ransac_parameters.time_axis)
    def _time_axis(value: int):
        plugin_ransac_parameters.time_axis.value = value

    @change_handler(plugin_ransac_parameters.defaults_params_button)
    def restore_prediction_parameters_defaults():
        for k, v in DEFAULTS_PRED_PARAMETERS.items():
            getattr(plugin_ransac_parameters, k).value = v

    @change_handler(plugin.defaults_model_button)
    def restore_model_defaults():
        for k, v in DEFAULTS_SEG_PARAMETERS.items():
            getattr(plugin, k).value = v

    # -> triggered by napari (if there are any open images on plugin launch)

    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        plugin.image.tooltip = (
            f"Shape: {get_data(image).shape, str(image.name)}"
        )

        # dimensionality of selected model: 2, 3, or None (unknown)
        ndim = get_data(image).ndim

        if (
            plugin.vollseg_model_type.value
            != DEFAULTS_MODEL["model_vollseg_none"]
        ):
            if model_selected_vollseg in model_vollseg_configs:
                config = model_vollseg_configs[model_selected_vollseg]
                ndim_model = config.get("n_dim")
        axes = None

        if ndim == 3:
            axes = "TYX"
            plugin.n_tiles.value = (1, 1, 1)
        elif ndim == 2 and ndim_model == 2:
            axes = "YX"
            plugin.n_tiles.value = (1, 1)

        else:
            raise NotImplementedError()

        if axes == plugin.axes.value:
            # make sure to trigger a changed event, even if value didn't actually change
            plugin.axes.changed(axes)
        else:
            plugin.axes.value = axes
        plugin.n_tiles.changed(plugin.n_tiles.value)

    # -> triggered by _image_change
    @change_handler(plugin.axes, plugin.vollseg_model_type, init=False)
    def _axes_change():
        value = plugin.axes.value
        image = plugin.image.value
        axes = plugin.axes.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            axes = axes_check_and_normalize(
                value, length=get_data(image).ndim, disallowed="S"
            )
            if (
                plugin.vollseg_model_type.value
                != DEFAULTS_MODEL["model_vollseg_none"]
            ):
                update_vollseg("image_axes", True, (axes, image, None))
        except ValueError as err:
            if (
                plugin.vollseg_model_type.value
                != DEFAULTS_MODEL["model_vollseg_none"]
            ):
                update_vollseg("image_axes", False, (value, image, err))
        # finally:
        # widgets_inactive(plugin.timelapse_opts, active=('T' in axes))

    # -> triggered by _image_change
    @change_handler(plugin.n_tiles, plugin.vollseg_model_type, init=False)
    def _n_tiles_change():
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            value = plugin.n_tiles.get_value()

            shape = get_data(image).shape
            try:
                value = tuple(value)
                len(value) == len(shape) or _raise(TypeError())
            except TypeError:
                raise ValueError(
                    f"must be a tuple/list of length {len(shape)}"
                )
            if not all(isinstance(t, int) and t >= 1 for t in value):
                raise ValueError("each value must be an integer >= 1")
            if (
                plugin.vollseg_model_type.value
                != DEFAULTS_MODEL["model_vollseg_none"]
            ):
                update_vollseg("n_tiles", True, (value, image, None))
        except (ValueError, SyntaxError) as err:
            if (
                plugin.vollseg_model_type.value
                != DEFAULTS_MODEL["model_vollseg_none"]
            ):
                update_vollseg("n_tiles", False, (None, image, err))

    # -------------------------------------------------------------------------

    return plugin
