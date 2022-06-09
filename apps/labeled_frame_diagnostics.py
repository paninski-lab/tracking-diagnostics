"""Analyze predictions on labeled frames.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- barplot of pixel errors for each model

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

from diagnostics.handler import ModelHandler
from diagnostics.metrics import pca_reprojection_error_per_keypoint
from diagnostics.streamlit import strip_cols_append_name, get_col_names, get_full_name
from diagnostics.streamlit import concat_dfs
from diagnostics.streamlit import compute_metric_per_dataset


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
    df_box = pd.concat(df_boxes)
    return df_box


st.title("Labeled Frame Diagnostics")

st.sidebar.header("Data Settings")
label_file: list = st.sidebar.file_uploader(
    "Choose CSV file with labeled data", accept_multiple_files=False
)
prediction_files: list = st.sidebar.file_uploader(
    "Choose one or more prediction CSV files", accept_multiple_files=True
)

n_cols = 3

if label_file is not None and len(prediction_files) > 0:  # otherwise don't try to proceed

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------

    # read dataframes into a dict with keys=filenames
    dframe_gt = pd.read_csv(label_file, header=[1, 2], index_col=0)

    dframes = {}
    for prediction_file in prediction_files:
        dframes[prediction_file.name] = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
        data_types = dframes[prediction_file.name].iloc[:, -1].unique()
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
    # user options
    # ---------------------------------------------------
    st.header("Select data to plot")

    keypoint_to_plot = st.selectbox(
        "Pick a single bodypart:",
        pd.Series([*keypoint_names, "mean", "ALL"]),
        key="bodypart_metric",
    )
    data_type = st.radio("Select data partition", data_types)

    # ---------------------------------------------------
    # plot metrics for all models
    # ---------------------------------------------------

    st.header("Compare multiple models")

    big_df_pix_error = compute_metric_per_dataset(
        dfs=dframes, metric='rmse', bodypart_names=keypoint_names, labels=dframe_gt)
    # TODO: compute other metrics

    # enumerate plotting options
    plot_type = st.selectbox("Pick a plot type:", ["box", "boxen", "bar", "violin", "strip"])
    plot_scale = st.radio("Select y-axis scale", ["linear", "log"])

    # filter data
    big_df_filtered = big_df_pix_error[big_df_pix_error.set == data_type]
    n_frames_per_dtype = big_df_filtered.shape[0] // len(prediction_files)

    # plot data
    title = '%s (%i %s frames)' % (keypoint_to_plot, n_frames_per_dtype, data_type)

    if plot_scale == "linear":
        log_y = False
    else:
        log_y = True

    if keypoint_to_plot == "ALL":

        sns.set_context("paper")

        df_box = get_df_box(big_df_filtered, keypoint_names, new_names)

        fig_box = sns.catplot(
            x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
            kind=plot_type, data=df_box, height=2)

        fig_box.set_xticklabels(rotation=45, ha='right')
        fig_box.fig.subplots_adjust(top=0.94)
        fig_box.fig.suptitle("All keypoints (%i %s frames)" % (n_frames_per_dtype, data_type))

        st.pyplot(fig_box)

    else:

        sns.set_context("paper")

        fig_box = plt.figure(figsize=(5, 5))

        if plot_type == "box":
            sns.boxplot(x="model_name", y=keypoint_to_plot, data=big_df_filtered)
        elif plot_type == "boxen":
            sns.boxenplot(x="model_name", y=keypoint_to_plot, data=big_df_filtered)
        elif plot_type == "bar":
            sns.barplot(x="model_name", y=keypoint_to_plot, data=big_df_filtered)
        elif plot_type == "violin":
            sns.violinplot(x="model_name", y=keypoint_to_plot, data=big_df_filtered)
        elif plot_type == "strip":
            sns.stripplot(x="model_name", y=keypoint_to_plot, data=big_df_filtered)
        else:
            raise NotImplementedError

        ax = fig_box.gca()
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        ax.set_xlabel("Model Name")
        ax.set_ylabel("Pixel Error")

        fig_box.subplots_adjust(top=0.95)
        fig_box.suptitle(title)

        st.pyplot(fig_box)

        # if plot_type == "box" or plot_type == "boxen":
        #     fig_box = px.box(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
        # elif plot_type == "violin":
        #     fig_box = px.violin(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
        # elif plot_type == "strip":
        #     fig_box = px.strip(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
        # # elif plot_type == "bar":
        # #     fig_box = px.bar(big_df_filtered, x="model_name", y=bodypart_error, log_y=log_y)
        # else:
        #     raise NotImplementedError
        # fig_width = 500
        # fig_height = 500
        # fig_box.update_layout(
        #     yaxis_title="Pixel Error", xaxis_title="Model Name", title=title,
        #     width=fig_width, height=fig_height)
        #
        # st.plotly_chart(fig_box)

    # ---------------------------------------------------
    # scatterplots
    # ---------------------------------------------------

    if len(prediction_files) > 0:

        st.header("Compare two models")
        model_0 = st.selectbox(
            "Model 0 (x-axis):", new_names, key="model_0")
        model_1 = st.selectbox(
            "Model 1 (y-axis):", [n for n in new_names if n != model_0], key="model_1")

        df_tmp0 = big_df_pix_error[big_df_pix_error.model_name == model_0]
        df_tmp1 = big_df_pix_error[big_df_pix_error.model_name == model_1]

        plot_scatter_scale = st.radio("Select axes scale", ["linear", "log"])
        if plot_scatter_scale == "linear":
            log_scatter = False
        else:
            log_scatter = True

        if keypoint_to_plot == "ALL":

            df_scatters = []
            for keypoint in keypoint_names:
                df_scatters.append(pd.DataFrame({
                    "img_file": df_tmp0.img_file[df_tmp0.set == data_type],
                    "keypoint": keypoint,
                    model_0: df_tmp0[keypoint][df_tmp0.set == data_type],
                    model_1: df_tmp1[keypoint][df_tmp1.set == data_type],
                }))
            df_scatter = pd.concat(df_scatters)

            fig_scatter = px.scatter(
                df_scatter,
                x=model_0, y=model_1,
                facet_col="keypoint", facet_col_wrap=n_cols,
                log_x=log_scatter, log_y=log_scatter,
                opacity=0.5,
                hover_data=['img_file'],
                # trendline="ols",
                title=title,
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
            )
            fig_width = 500
            fig_height = 500

        mn = np.min(df_scatter[[model_0, model_1]].min(skipna=True).to_numpy())
        mx = np.max(df_scatter[[model_0, model_1]].max(skipna=True).to_numpy())
        trace = go.Scatter(x=[mn, mx], y=[mn, mx], line_color="black", mode="lines")
        trace.update(legendgroup="trendline", showlegend=False)
        fig_scatter.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
        fig_scatter.update_layout(
            xaxis_title=model_0, yaxis_title=model_1, title=title,
            width=fig_width, height=fig_height)
        fig_scatter.update_traces(marker={'size': 5})
        st.plotly_chart(fig_scatter)
