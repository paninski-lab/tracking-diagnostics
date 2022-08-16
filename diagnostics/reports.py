"""Output a collection of plots that parallel those provided by the streamlit apps."""
from typing import List

import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from typing import List, Dict, Tuple, Optional
import yaml

from lightning_pose.losses.losses import PCALoss
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    get_imgaug_transform, get_dataset, get_data_module, get_loss_factories,
)

from diagnostics.handler import ModelHandler
from diagnostics.metrics import rmse
from diagnostics.metrics import pca_reprojection_error_per_keypoint
from diagnostics.video import make_labeled_video
from diagnostics.visualizations import get_y_label, make_seaborn_catplot, make_plotly_scatterplot
from diagnostics.visualizations import plot_traces


pix_error_key = "pixel error"
conf_error_key = "confidence"
temp_norm_error_key = "temporal norm"
pcamv_error_key = "pca multiview"
pcasv_error_key = "pca singleview"


class ReportGenerator:
    """Generate a report that includes plots and videos.

    Examples
    --------
    # Note: this class cannot be imported into a notebook, it must be run in a script

    # initialize object for labeled frames
    reporter = GenerateReport(
        prediction_files=["/path/to/preds0.csv", "/path/to/preds1.csv"],
        model_names=["model_0", "model_1"],
        cfg_file="/path/to/config.yaml",
        label_file="/path/to/labels.csv",
    )
    # create report
    report_dir = reporter.generate_report(save_dir="/path/to/report_dir", format="pdf")

    """

    def __init__(
            self,
            prediction_files: list,
            model_names: list,
            cfg_file: Optional[str] = None,
            label_file: Optional[str] = None,
            video_file: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        prediction_files
            list of absolute paths to prediction csv files (either frame or video)
        model_names
            list of corresponding model names
        cfg
            Lightning Pose config file
        label_file
            absolute path to labeled data
        video_file
            absolute path to video file; if this is not None, a labeled video is created during
            report generation

        """
        assert len(prediction_files) == len(model_names), \
            "Must provide one model name for each prediction file"

        # save inupts
        self.prediction_files = prediction_files
        self.model_names = model_names
        self.cfg_file = cfg_file
        self.label_file = label_file
        self.video_file = video_file

        self.cfg = None
        self.is_video = True if label_file is None else False

        # store data in dfs
        self.dframe_gt = None  # labeled data
        self.dframes = {}  # raw dataframes with predictions
        self.df_concat = None  # concatenation that includes model names
        self.df_metrics = None  # store computed metrics
        self.keypoint_names = None

        # process data
        self._load_data_make_dfs()

        self.df_metrics = build_metrics_df(
            dframes=self.dframes, keypoint_names=self.keypoint_names, is_video=self.is_video,
            cfg=self.cfg)
        self.metric_options = list(self.df_metrics.keys())

    def _load_data_make_dfs(self):

        # load labels if present
        if self.label_file is not None:
            self.dframe_gt = pd.read_csv(self.label_file, header=[1, 2], index_col=0)

        # load predictions, either labeled frames or videos
        for model_name, prediction_file in zip(self.model_names, self.prediction_files):
            self.dframes[model_name] = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
        self.df_concat, self.keypoint_names = concat_dfs(self.dframes)

        # load config if present
        if self.cfg_file is not None:
            self.cfg = DictConfig(yaml.safe_load(open(self.cfg_file)))

    def generate_report(
            self,
            save_dir: str,
            format: str = "pdf",
            box_kwargs: dict = {},
            scatter_kwargs: dict = {},
            trace_kwargs: dict = {},
            video_kwargs: dict = {},
            make_video_per_keypoint: bool = False,
    ) -> str:
        """

        Parameters
        ----------
        save_dir
            report will be saved in `save_dir/litpose-report-[video/labeled]_[date_time]`
        format
            "pdf" | "png"
        box_kwargs
            "plot_type": "boxen" | "box" | "bar" | "violin" | "strip"
            "plot_scale": "log" | "linear"
            "data_type" (labeled only): "train" | "validation" | "test" | "unused"
        scatter_kwargs
            "plot_scale": "log" | "linear"
            "data_type" (labeled only): "train" | "validation" | "test" | "unused"
            "model_0": <model_str>,
            "model_1": <model_str>,
        trace_kwargs
            "model_names": list of model names as strs
        video_kwargs
            "likelihood_thresh": float
            "max_frames": int
            "markersize": int
            "framerate": float
            "height": float (inches)
        make_video_per_keypoint
            False will make a single video with all keypoints (if video_file was passed to
            constructor). True will make the all-keypoint video as well as one video for each
            keypoint.

        Returns
        -------
        save directory

        """

        save_dir_full = self.generate_save_dir(save_dir, is_video=self.is_video)

        if self.is_video:
            if "model_names" not in trace_kwargs:
                trace_kwargs["model_names"] = self.model_names
            generate_report_video(
                df_traces=self.df_concat,
                df_metrics=self.df_metrics,
                keypoint_names=self.keypoint_names,
                save_dir=save_dir_full,
                format=format,
                box_kwargs=box_kwargs,
                trace_kwargs=trace_kwargs,
                video_kwargs=video_kwargs,
                savefig_kwargs={},
                video_file=self.video_file,
                make_video_per_keypoint=make_video_per_keypoint,
            )
        else:
            if "model_0" not in scatter_kwargs:
                scatter_kwargs["model_0"] = self.model_names[0]
            if "model_1" not in scatter_kwargs:
                scatter_kwargs["model_1"] = self.model_names[1]
            generate_report_labeled(
                df_metrics=self.df_metrics,
                keypoint_names=self.keypoint_names,
                model_names=self.model_names,
                save_dir=save_dir_full,
                format=format,
                box_kwargs=box_kwargs,
                scatter_kwargs=scatter_kwargs,
                savefig_kwargs={},
            )
        return save_dir_full

    @staticmethod
    def generate_save_dir(base_save_dir: str, is_video: bool):
        run_date_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        if is_video:
            save_dir = os.path.join(base_save_dir, "litpose-report-video_%s" % run_date_time)
        else:
            save_dir = os.path.join(base_save_dir, "litpose-report-labeled_%s" % run_date_time)
        return save_dir


def update_kwargs_dict_with_defaults(kwargs_new, kwargs_default):
    for key, val in kwargs_default.items():
        if key not in kwargs_new:
            kwargs_new[key] = val
    return kwargs_new


def generate_report_labeled(
        df_metrics,
        keypoint_names,
        model_names,
        save_dir,
        format="pdf",
        box_kwargs={},
        scatter_kwargs={},
        savefig_kwargs={}
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse inputs
    box_kwargs_default = {
        "plot_type": "boxen",
        "plot_scale": "log",
        "data_type": "test",
    }
    box_kwargs = update_kwargs_dict_with_defaults(box_kwargs, box_kwargs_default)

    scatter_kwargs_default = {
        "plot_scale": "log",
        "data_type": "test",
        "model_0": "model_0",
        "model_1": "model_1",
    }
    scatter_kwargs = update_kwargs_dict_with_defaults(scatter_kwargs, scatter_kwargs_default)

    metrics_to_plot = df_metrics.keys()

    # ---------------------------------------------------
    # plot metrics for all models (catplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        df_filtered = df_metrics[metric_to_plot][
            df_metrics[metric_to_plot].set == box_kwargs["data_type"]]
        n_frames_per_dtype = df_filtered.shape[0] // len(df_filtered.model_name.unique())

        y_label = get_y_label(metric_to_plot)

        # plot ALL data
        n_cols = 3
        df_box = get_df_box(df_filtered, keypoint_names, model_names)
        sns.set_context("paper")
        title = "All keypoints (%i %s frames)" % (n_frames_per_dtype, box_kwargs["data_type"])
        fig_box = sns.catplot(
            x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
            kind=box_kwargs["plot_type"], data=df_box, height=2)
        fig_box.set_axis_labels("Model Name", y_label)
        fig_box.set_xticklabels(rotation=45, ha="right")
        fig_box.set(yscale=box_kwargs["plot_scale"])
        fig_box.fig.subplots_adjust(top=0.94)
        fig_box.fig.suptitle(title)
        save_file = os.path.join(save_dir, "barplot-all_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

        # plot mean data
        keypoint_to_plot = "mean"
        title = 'Mean keypoints (%i %s frames)' % (n_frames_per_dtype, box_kwargs["data_type"])
        log_y = False if box_kwargs["plot_scale"] == "linear" else True
        make_seaborn_catplot(
            x="model_name", y=keypoint_to_plot, data=df_filtered, x_label="Model Name",
            y_label=y_label, title=title, log_y=log_y, plot_type=box_kwargs["plot_type"])
        save_file = os.path.join(save_dir, "barplot-mean_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

    # ---------------------------------------------------
    # plot metrics for a pair of models (scatterplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        model_0 = scatter_kwargs["model_0"]
        model_1 = scatter_kwargs["model_1"]

        df_tmp0 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_0]
        df_tmp1 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_1]

        y_label = get_y_label(metric_to_plot)

        # ---------------
        # plot ALL data
        # ---------------
        n_cols = 3
        df_scatter = get_df_scatter(
            df_tmp0, df_tmp1, scatter_kwargs["data_type"], [model_0, model_1], keypoint_names)
        title = "All keypoints (%i %s frames)" % (n_frames_per_dtype, scatter_kwargs["data_type"])
        fig_scatter = make_plotly_scatterplot(
            model_0=model_0, model_1=model_1, df=df_scatter,
            metric_name=y_label, title=title,
            axes_scale=scatter_kwargs["plot_scale"],
            facet_col="keypoint", n_cols=n_cols,
            fig_height=300 * np.ceil(len(keypoint_names) / n_cols), fig_width=900,
        )
        save_file = os.path.join(save_dir, "scatterplot-all_%s.%s" % (metric_to_plot, format))
        fig_scatter.write_image(save_file)

        # ---------------
        # plot mean data
        # ---------------
        keypoint_to_plot = "mean"
        df_scatter = pd.DataFrame({
            model_0: df_tmp0[keypoint_to_plot][df_tmp0.set == scatter_kwargs["data_type"]],
            model_1: df_tmp1[keypoint_to_plot][df_tmp1.set == scatter_kwargs["data_type"]],
        })
        fig_scatter = make_plotly_scatterplot(
            model_0=model_0, model_1=model_1, df=df_scatter,
            metric_name=y_label, title=title,
            axes_scale=scatter_kwargs["plot_scale"],
            fig_height=500, fig_width=500,
        )
        save_file = os.path.join(save_dir, "scatterplot-mean_%s.%s" % (metric_to_plot, format))
        fig_scatter.write_image(save_file)


def generate_report_video(
        df_traces,
        df_metrics,
        keypoint_names,
        save_dir,
        format="pdf",
        box_kwargs={},
        trace_kwargs={},
        savefig_kwargs={},
        video_kwargs={},
        video_file=None,
        make_video_per_keypoint=False,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse inputs
    box_kwargs_default = {
        "plot_type": "boxen",
        "plot_scale": "log",
    }
    box_kwargs = update_kwargs_dict_with_defaults(box_kwargs, box_kwargs_default)

    trace_kwargs_default = {
        "model_names": [],
    }
    trace_kwargs = update_kwargs_dict_with_defaults(trace_kwargs, trace_kwargs_default)

    video_kwargs_default = {
        "likelihood_thresh": 0.05,
        "max_frames": 1000,
        "markersize": 6,
        "framerate": 20,
        "height": 4,
    }
    video_kwargs = update_kwargs_dict_with_defaults(video_kwargs, video_kwargs_default)

    metrics_to_plot = df_metrics.keys()

    # ---------------------------------------------------
    # plot metrics for all models (catplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        x_label = "Model Name"
        y_label = get_y_label(metric_to_plot)

        # plot ALL data
        df_tmp = df_metrics[metric_to_plot].melt(id_vars="model_name")
        df_tmp = df_tmp.rename(columns={"variable": "keypoint"})
        fig_box = sns.catplot(
            data=df_tmp, x="model_name", y="value", col="keypoint", col_wrap=3,
            kind=box_kwargs["plot_type"]
        )
        fig_box.set(yscale=box_kwargs["plot_scale"])
        save_file = os.path.join(save_dir, "barplot-all_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

        # plot mean data
        log_y = False if box_kwargs["plot_scale"] == "linear" else True
        make_seaborn_catplot(
            x="model_name", y="mean", data=df_metrics[metric_to_plot], log_y=log_y,
            x_label=x_label, y_label=y_label, title="Average over all keypoints",
            plot_type=box_kwargs["plot_type"])
        save_file = os.path.join(save_dir, "barplot-mean_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

    # ---------------------------------------------------
    # plot traces
    # ---------------------------------------------------
    for keypoint in keypoint_names:

        cols = get_col_names(keypoint, "x", trace_kwargs["model_names"])
        fig_traces = plot_traces(df_metrics, df_traces, cols)
        save_file = os.path.join(save_dir, "traces-%s.%s" % (keypoint, format))
        fig_traces.write_image(save_file)

    # ---------------------------------------------------
    # make labeled video(s)
    # ---------------------------------------------------
    if video_file is not None:

        cap = cv2.VideoCapture(video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # extract x, y, likelihoods for each model
        points = []
        for model_name in trace_kwargs["model_names"]:
            points_model = {}
            for keypoint in keypoint_names:
                tmp = []
                for val in ["x", "y", "likelihood"]:
                    tmp.append(
                        df_traces[get_full_name(keypoint, val, model_name)].to_numpy()[:, None])
                points_model[keypoint] = np.concatenate(tmp, axis=1)
            points.append(points_model)

        # make single video with all keypoints
        save_file = os.path.join(save_dir, "%s_all.mp4" % video_name)
        make_labeled_video(save_file, cap, points, trace_kwargs["model_names"], **video_kwargs)

        # make individual video for each keypoint
        if make_video_per_keypoint:
            for keypoint in keypoint_names:
                save_file = os.path.join(save_dir, "%s_%s.mp4" % (video_name, keypoint))
                points_single = [{keypoint: points_tmp[keypoint]} for points_tmp in points]
                make_labeled_video(
                    save_file, cap, points_single, trace_kwargs["model_names"], **video_kwargs)


# ----------------------------------------------
# manipulate dataframes
# ----------------------------------------------
@st.cache
def concat_dfs(dframes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    counter = 0
    for model_name, dframe in dframes.items():
        if counter == 0:
            df_concat = dframe.copy()
            # base_colnames = list(df_concat.columns.levels[0])  # <-- sorts names, bad!
            base_colnames = list([c[0] for c in df_concat.columns[1::3]])
            df_concat = strip_cols_append_name(df_concat, model_name)
        else:
            df = strip_cols_append_name(dframe.copy(), model_name)
            df_concat = pd.concat([df_concat, df], axis=1)
        counter += 1
    return df_concat, base_colnames


@st.cache
def get_df_box(df_orig, keypoint_names, model_names):
    df_boxes = []
    for keypoint in keypoint_names:
        for model_curr in model_names:
            tmp_dict = {
                "keypoint": keypoint,
                "metric": "Pixel error",
                "value": df_orig[df_orig.model_name == model_curr][keypoint],
                "model_name": model_curr,
            }
            df_boxes.append(pd.DataFrame(tmp_dict))
    return pd.concat(df_boxes)


@st.cache
def get_df_scatter(df_0, df_1, data_type, model_names, keypoint_names):
    df_scatters = []
    for keypoint in keypoint_names:
        df_scatters.append(pd.DataFrame({
            "img_file": df_0.img_file[df_0.set == data_type],
            "keypoint": keypoint,
            model_names[0]: df_0[keypoint][df_0.set == data_type],
            model_names[1]: df_1[keypoint][df_1.set == data_type],
        }))
    return pd.concat(df_scatters)


def get_col_names(keypoint: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(keypoint, coordinate, model) for model in models]


def strip_cols_append_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col + "_" + name for col in df.columns.values]
    return df


def get_full_name(keypoint: str, coordinate: str, model: str) -> str:
    return "_".join([keypoint, coordinate, model])


# ----------------------------------------------
# compute metrics
# ----------------------------------------------
@st.cache
def build_metrics_df(dframes, keypoint_names, is_video, cfg=None, dframe_gt=None) -> dict:

    df_metrics = dict()

    # confidence
    df_metrics[conf_error_key] = compute_metric_per_dataset(
        dfs=dframes, metric="confidence", keypoint_names=keypoint_names)

    if is_video:
        # temporal norm
        df_metrics[temp_norm_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="temporal_norm", keypoint_names=keypoint_names)
    else:
        # pixel error
        if dframe_gt is not None:
            df_metrics[pix_error_key] = compute_metric_per_dataset(
                dfs=dframes, metric="rmse", keypoint_names=keypoint_names, labels=dframe_gt)

    # pca multiview
    if cfg is not None and cfg.data.get("mirrored_column_matches", None):
        cfg_pcamv = cfg.copy()
        cfg_pcamv.model.losses_to_use = ["pca_multiview"]
        pcamv_loss = build_pca_loss_object(cfg_pcamv)
        df_metrics[pcamv_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="pca_mv", keypoint_names=keypoint_names, cfg=cfg_pcamv,
            pca_loss=pcamv_loss)

    # pca singleview
    if cfg is not None and cfg.data.get("columns_for_singleview_pca", None):
        cfg_pcasv = cfg.copy()
        cfg_pcasv.model.losses_to_use = ["pca_singleview"]
        pcasv_loss = build_pca_loss_object(cfg_pcasv)
        df_metrics[pcasv_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="pca_sv", keypoint_names=keypoint_names, cfg=cfg_pcasv,
            pca_loss=pcasv_loss)

    return df_metrics


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def compute_metric_per_dataset(
    dfs: Dict[str, pd.DataFrame], metric: str, keypoint_names: List[str], **kwargs
) -> pd.DataFrame:

    # colnames = [*keypoint_names, "model_name"]
    # if "labels" in kwargs:
    #     colnames += ["imag_file"]
    # concat_df = pd.DataFrame(columns=colnames)
    concat_dfs = []
    for model_name, df in dfs.items():
        if metric == "rmse":
            df_ = compute_pixel_error(df, keypoint_names, model_name, **kwargs)
        elif metric == "temporal_norm":
            df_ = compute_temporal_norms(df, keypoint_names, model_name, **kwargs)
        elif metric == "pca_mv" or metric == "pca_sv":
            df_ = compute_pca_reprojection_error(df, keypoint_names, model_name, **kwargs)
        elif metric == "confidence":
            df_ = compute_confidence(df, keypoint_names, model_name, **kwargs)
        else:
            raise NotImplementedError("%s is not a supported metric" % metric)
        # concat_df = pd.concat([concat_df, df_.reset_index(inplace=True, drop=False)], axis=0)
        # concat_df = pd.concat([concat_df, df_], axis=0, ignore_index=False)
    # return concat_df
        concat_dfs.append(df_)
    return pd.concat(concat_dfs)


@st.cache
def compute_pixel_error(
    df: pd.DataFrame, keypoint_names: List[str], model_name: str, labels: pd.DataFrame,
) -> pd.DataFrame:

    # shape (samples, n_keypoints, 2)
    keypoints_true = labels.to_numpy().reshape(labels.shape[0], -1, 2)

    # shape (samples, n_keypoints, 2)
    tmp = df.iloc[:, :-1].to_numpy().reshape(df.shape[0], -1, 3)  # remove "set" column
    keypoints_pred = tmp[:, :, :2]

    set = df.iloc[:, -1].to_numpy()

    results = rmse(keypoints_true, keypoints_pred)

    # collect results
    df_ = pd.DataFrame(columns=keypoint_names)
    for c, col in enumerate(keypoint_names):  # loop over keypoints
        df_[col] = results[:, c]
    df_["model_name"] = model_name
    df_["mean"] = df_[keypoint_names[:-1]].mean(axis=1)
    df_["set"] = set
    df_["img_file"] = labels.index

    return df_


@st.cache
def compute_confidence(
        df: pd.DataFrame, keypoint_names: List[str], model_name: str, **kwargs
) -> pd.DataFrame:

    if df.shape[1] % 3 == 1:
        # get rid of "set" column if present
        tmp = df.iloc[:, :-1].to_numpy().reshape(df.shape[0], -1, 3)
        set = df.iloc[:, -1].to_numpy()
    else:
        tmp = df.to_numpy().reshape(df.shape[0], -1, 3)
        set = None

    results = tmp[:, :, 2]

    # collect results
    df_ = pd.DataFrame(columns=keypoint_names)
    for c, col in enumerate(keypoint_names):  # loop over keypoints
        df_[col] = results[:, c]
    df_["model_name"] = model_name
    df_["mean"] = df_[keypoint_names[:-1]].mean(axis=1)
    if set is not None:
        df_["set"] = set
        df_["img_file"] = df.index

    return df_


@st.cache
def compute_temporal_norms(
    df: pd.DataFrame, keypoint_names: List[str], model_name: str, **kwargs
) -> pd.DataFrame:
    # compute the norm just for one dataframe
    df_norms = pd.DataFrame(columns=keypoint_names)
    diffs = df.diff(periods=1)  # not using .abs
    for col in keypoint_names:  # loop over keypoints
        df_norms[col] = diffs[col][["x", "y"]].apply(
            np.linalg.norm, axis=1
        )  # norm of the difference for that keypoint
    df_norms["model_name"] = model_name
    df_norms["mean"] = df_norms[keypoint_names[:-1]].mean(axis=1)
    return df_norms


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def compute_pca_reprojection_error(
    df: pd.DataFrame, keypoint_names: List[str], model_name: str, cfg: dict, pca_loss: PCALoss,
) -> pd.DataFrame:

    # TODO: copied from diagnostics.handler.Handler::compute_metric; figure out a way to share
    if df.shape[1] % 3 == 1:
        # get rid of "set" column if present
        tmp = df.iloc[:, :-1].to_numpy().reshape(df.shape[0], -1, 3)
        set = df.iloc[:, -1].to_numpy()
    else:
        tmp = df.to_numpy().reshape(df.shape[0], -1, 3)
        set = None
    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)
    keypoints_pred = ModelHandler.resize_keypoints(cfg, keypoints_pred=keypoints_pred)
    original_dims = keypoints_pred.shape

    if pca_loss.loss_name == "pca_multiview":

        mirrored_column_matches = pca_loss.pca.mirrored_column_matches
        # adding a reshaping below since the loss class expects a single last dim with
        # num_keypoints*2
        results_raw = pca_reprojection_error_per_keypoint(
            pca_loss, keypoints_pred=keypoints_pred.reshape(keypoints_pred.shape[0], -1))
        results_raw = results_raw.reshape(
            -1,
            len(mirrored_column_matches[0]),
            len(mirrored_column_matches),
        )  # batch X num_used_keypoints X num_views

        # next, put this back into a full keypoints pred arr
        results = np.nan * np.zeros((original_dims[0], original_dims[1]))
        for c, cols in enumerate(mirrored_column_matches):
            results[:, cols] = results_raw[:, :, c]  # just the columns belonging to view c

    elif pca_loss.loss_name == "pca_singleview":

        pca_cols = pca_loss.pca.columns_for_singleview_pca
        results_ = pca_reprojection_error_per_keypoint(
            pca_loss, keypoints_pred=keypoints_pred.reshape(keypoints_pred.shape[0], -1))

        # next, put this back into a full keypoints pred arr
        results = np.nan * np.zeros((original_dims[0], original_dims[1]))
        results[:, pca_cols] = results_

    # collect results
    df_rpe = pd.DataFrame(columns=keypoint_names)
    for c, col in enumerate(keypoint_names):  # loop over keypoints
        df_rpe[col] = results[:, c]
    df_rpe["model_name"] = model_name
    df_rpe["mean"] = df_rpe[keypoint_names[:-1]].mean(axis=1)
    if set is not None:
        df_rpe["set"] = set
        df_rpe["img_file"] = df.index

    return df_rpe


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def build_pca_loss_object(cfg):
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
    data_module.setup()
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)
    pca_loss = loss_factories["unsupervised"].loss_instance_dict[cfg.model.losses_to_use[0]]
    return pca_loss
