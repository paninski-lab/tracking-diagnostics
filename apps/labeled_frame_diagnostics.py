"""Analyze predictions on labeled frames.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- plot of a selected metric (e.g. pixel errors) for each model (bar/box/violin/etc)
- scatterplot of a selected metric between two models

to run from command line:
> streamlit run /path/to/labeled_frame_diagnostics.py

"""

import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from omegaconf import DictConfig
import os
from typing import List, Dict, Tuple, Optional
import yaml

from lightning_pose.losses.losses import PCALoss
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    get_imgaug_transform, get_dataset, get_data_module, get_loss_factories,
)

from diagnostics.streamlit import build_pca_loss_object
from diagnostics.streamlit import concat_dfs
from diagnostics.streamlit import compute_metric_per_dataset
from diagnostics.streamlit import make_seaborn_catplot


# TODO
# - add new metrics (pca)
# - refactor df making
# - save as pdf / eps
# - show image on hover?


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
def get_df_scatter(df_0, df_1, data_type):
    df_scatters = []
    for keypoint in keypoint_names:
        df_scatters.append(pd.DataFrame({
            "img_file": df_0.img_file[df_0.set == data_type],
            "keypoint": keypoint,
            model_0: df_0[keypoint][df_0.set == data_type],
            model_1: df_1[keypoint][df_1.set == data_type],
        }))
    return pd.concat(df_scatters)


st.title("Labeled Frame Diagnostics")

st.sidebar.header("Data Settings")
label_file: list = st.sidebar.file_uploader(
    "Choose CSV file with labeled data", accept_multiple_files=False
)
prediction_files: list = st.sidebar.file_uploader(
    "Choose one or more prediction CSV files", accept_multiple_files=True
)

# col wrap when plotting results from all keypoints
n_cols = 3

# metrics to plot
pix_error_key = "pixel error"
pcamv_error_key = "pca multiview"
pcasv_error_key = "pca singleview"

metric_options = [pix_error_key]

if label_file is not None and len(prediction_files) > 0:  # otherwise don't try to proceed

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------

    # read dataframes into a dict with keys=filenames
    dframe_gt = pd.read_csv(label_file, header=[1, 2], index_col=0)

    dframes = {}
    for prediction_file in prediction_files:
        if prediction_file.name in dframes.keys():
            # append new integer to duplicate filenames
            idx = 0
            new_name = "%s_0" % prediction_file.name
            while new_name in dframes.keys():
                idx += 1
                new_name = "%s_%i" % (prediction_file.name, idx)
            filename = new_name
        else:
            filename = prediction_file.name
        dframe = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
        dframes[filename] = dframe
        data_types = dframe.iloc[:, -1].unique()
        # if dframes[prediction_file.name].keys()[-1][0] != "set":
        #     raise ValueError("Final column of %s must use \"set\" header" % prediction_file.name)

    # edit modelnames if desired, to simplify plotting
    st.sidebar.write("Model display names (editable)")
    new_names = []
    og_names = list(dframes.keys())
    for name in og_names:
        new_name = st.sidebar.text_input(label="", value=name)
        new_names.append(new_name)

    # change dframes key names to new ones
    for n_name, o_name in zip(new_names, og_names):
        dframes[n_name] = dframes.pop(o_name)

    # concat dataframes, collapsing hierarchy and making df fatter.
    df_concat, keypoint_names = concat_dfs(dframes)

    # ---------------------------------------------------
    # compute metrics
    # ---------------------------------------------------

    uploaded_cfg: str = st.sidebar.file_uploader(
        "Select data config yaml (optional, for pca losses)", accept_multiple_files=False
    )
    if uploaded_cfg is not None:
        cfg = DictConfig(yaml.safe_load(uploaded_cfg))
    else:
        cfg = None

    big_df = {}
    big_df[pix_error_key] = compute_metric_per_dataset(
        dfs=dframes, metric="rmse", keypoint_names=keypoint_names, labels=dframe_gt)
    if cfg is not None and cfg.data.get("mirrored_column_matches", None):
        cfg_pcamv = cfg.copy()
        cfg_pcamv.model.losses_to_use = ["pca_multiview"]
        pcamv_loss = build_pca_loss_object(cfg_pcamv)
        big_df[pcamv_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="pca_mv", keypoint_names=keypoint_names, cfg=cfg_pcamv,
            pca_loss=pcamv_loss)
        metric_options += [pcamv_error_key]
    if cfg is not None and cfg.data.get("columns_for_singleview_pca", None):
        cfg_pcasv = cfg.copy()
        cfg_pcasv.model.losses_to_use = ["pca_singleview"]
        pcasv_loss = build_pca_loss_object(cfg_pcasv)
        big_df[pcasv_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="pca_sv", keypoint_names=keypoint_names, cfg=cfg_pcasv,
            pca_loss=pcasv_loss)
        metric_options += [pcasv_error_key]

    # ---------------------------------------------------
    # user options
    # ---------------------------------------------------
    st.header("Select data to plot")

    # choose from individual keypoints, their mean, or all at once
    keypoint_to_plot = st.selectbox(
        "Select a keypoint:", ["mean", "ALL", *keypoint_names], key="keypoint")

    # choose which metric to plot
    metric_to_plot = st.selectbox("Select a metric:", metric_options, key="metric")
    if metric_to_plot == pix_error_key:
        y_label = "Pixel Error"
    elif metric_to_plot == pcamv_error_key:
        y_label = "Multiview PCA Reproj Error (pix)"
    elif metric_to_plot == pcasv_error_key:
        y_label = "Singleview PCA Reproj Error (pix)"

    # choose data split - train/val/test/unused
    data_type = st.selectbox("Select data partition:", data_types, key="data partition")

    # ---------------------------------------------------
    # plot metrics for all models
    # ---------------------------------------------------

    st.header("Compare multiple models")

    # enumerate plotting options
    plot_type = st.selectbox("Pick a plot type:", ["boxen", "box", "bar", "violin", "strip"])
    plot_scale = st.radio("Select y-axis scale", ["linear", "log"])

    # filter data
    big_df_filtered = big_df[metric_to_plot][big_df[metric_to_plot].set == data_type]
    n_frames_per_dtype = big_df_filtered.shape[0] // len(prediction_files)

    # plot data
    title = '%s (%i %s frames)' % (keypoint_to_plot, n_frames_per_dtype, data_type)

    log_y = False if plot_scale == "linear" else True

    if keypoint_to_plot == "ALL":

        df_box = get_df_box(big_df_filtered, keypoint_names, new_names)
        sns.set_context("paper")
        fig_box = sns.catplot(
            x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
            kind=plot_type, data=df_box, height=2)
        fig_box.set_axis_labels("Model Name", y_label)
        fig_box.set_xticklabels(rotation=45, ha="right")
        fig_box.fig.subplots_adjust(top=0.94)
        fig_box.fig.suptitle("All keypoints (%i %s frames)" % (n_frames_per_dtype, data_type))
        st.pyplot(fig_box)

    else:

        fig_box = make_seaborn_catplot(
            x="model_name", y=keypoint_to_plot, data=big_df_filtered, x_label="Model Name",
            y_label=y_label, title=title, log_y=log_y, plot_type=plot_type)
        st.pyplot(fig_box)

    # ---------------------------------------------------
    # scatterplots
    # ---------------------------------------------------

    st.header("Compare two models")
    model_0 = st.selectbox(
        "Model 0 (x-axis):", new_names, key="model_0")
    model_1 = st.selectbox(
        "Model 1 (y-axis):", [n for n in new_names if n != model_0], key="model_1")

    df_tmp0 = big_df[metric_to_plot][big_df[metric_to_plot].model_name == model_0]
    df_tmp1 = big_df[metric_to_plot][big_df[metric_to_plot].model_name == model_1]

    plot_scatter_scale = st.radio("Select axes scale", ["linear", "log"])
    log_scatter = False if plot_scatter_scale == "linear" else True

    xlabel_ = "%s (%s)" % (y_label, model_0)
    ylabel_ = "%s (%s)" % (y_label, model_1)

    if keypoint_to_plot == "ALL":

        df_scatter = get_df_scatter(df_tmp0, df_tmp1, data_type)

        fig_scatter = px.scatter(
            df_scatter,
            x=model_0, y=model_1,
            facet_col="keypoint", facet_col_wrap=n_cols,
            log_x=log_scatter, log_y=log_scatter,
            opacity=0.5,
            hover_data=['img_file'],
            # trendline="ols",
            title=title,
            labels={model_0: xlabel_, model_1: ylabel_},
        )

        fig_width = 900
        fig_height = 300 * np.ceil(len(keypoint_names) / n_cols)

    else:

        df_scatter = pd.DataFrame({
            model_0: df_tmp0[keypoint_to_plot][df_tmp0.set == data_type],
            model_1: df_tmp1[keypoint_to_plot][df_tmp1.set == data_type],
            "img_file": df_tmp0.img_file[df_tmp0.set == data_type]
        })
        fig_scatter = px.scatter(
            df_scatter,
            x=model_0, y=model_1,
            log_x=log_scatter, log_y=log_scatter,
            opacity=0.5,
            hover_data=['img_file'],
            # trendline="ols",
            title=title,
            labels={model_0: xlabel_, model_1: ylabel_},
        )
        fig_width = 500
        fig_height = 500

    mn = np.min(df_scatter[[model_0, model_1]].min(skipna=True).to_numpy())
    mx = np.max(df_scatter[[model_0, model_1]].max(skipna=True).to_numpy())
    trace = go.Scatter(x=[mn, mx], y=[mn, mx], line_color="black", mode="lines")
    trace.update(legendgroup="trendline", showlegend=False)
    fig_scatter.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
    fig_scatter.update_layout(title=title, width=fig_width, height=fig_height)
    fig_scatter.update_traces(marker={'size': 5})
    st.plotly_chart(fig_scatter)
