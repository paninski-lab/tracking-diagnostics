"""Analyze predictions on video data.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- time series/likelihoods of a selected keypoint (x or y coord) for each model
- boxplot/histogram of temporal norms for each model
- boxplot/histogram of multiview pca reprojection errors for each model

to run from command line:
> streamlit run /path/to/video_diagnostics.py

"""

# from email.mime import base
# from urllib.parse import _NetlocResultMixinBase
# from grpc import dynamic_ssl_server_credentials
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
from typing import List, Dict, Tuple, Optional
import yaml

from diagnostics.streamlit import get_col_names
from diagnostics.streamlit import concat_dfs
from diagnostics.streamlit import compute_metric_per_dataset
from diagnostics.streamlit import build_pca_loss_object


def make_seaborn_catplot(
        x, y, data, x_label, y_label, title, log_y=False, plot_type="box", figsize=(5, 5)):
    sns.set_context("paper")
    fig = plt.figure(figsize=figsize)
    if plot_type == "box":
        sns.boxplot(x=x, y=y, data=data)
    elif plot_type == "boxen":
        sns.boxenplot(x=x, y=y, data=data)
    elif plot_type == "bar":
        sns.barplot(x=x, y=y, data=data)
    elif plot_type == "violin":
        sns.violinplot(x=x, y=y, data=data)
    elif plot_type == "strip":
        sns.stripplot(x=x, y=y, data=data)
    else:
        raise NotImplementedError
    ax = fig.gca()
    ax.set_yscale("log") if log_y else ax.set_yscale("linear")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title)
    return fig


def make_plotly_catplot(x, y, data, x_label, y_label, title, plot_type="box"):

    if plot_type == "box":
        fig = px.box(data, x=x, y=y)
        fig.update_layout(yaxis_title=y_label, xaxis_title=x_label, title=title)
    elif plot_type == "hist":
        fig = px.histogram(
            data, x=x, color="model_name", marginal="rug", barmode="overlay",
        )
        fig.update_layout(yaxis_title=y_label, xaxis_title=x_label, title=title)

    return fig


st.title("Video Diagnostics")

st.sidebar.header("Data Settings")
uploaded_files: list = st.sidebar.file_uploader(
    "Choose one or more CSV files", accept_multiple_files=True
)

if len(uploaded_files) > 0:  # otherwise don't try to proceed

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------

    # read dataframes into a dict with keys=filenames
    dframes = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.name in dframes.keys():
            # append new integer to duplicate filenames
            idx = 0
            new_name = "%s_0" % uploaded_file.name
            while new_name in dframes.keys():
                idx += 1
                new_name = "%s_%i" % (uploaded_file.name, idx)
            filename = new_name
        else:
            filename = uploaded_file.name
        dframes[filename] = pd.read_csv(uploaded_file, header=[1, 2], index_col=0)

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
    df_concat, bodypart_names = concat_dfs(dframes)

    # ---------------------------------------------------
    # plot traces
    # ---------------------------------------------------

    st.header("Trace diagnostics")

    # display_head = st.checkbox("Display trace DataFrame")
    # if display_head:
    #     st.write("Concatenated Dataframe:")
    #     st.write(df_concat.head())

    models = st.multiselect(
        "Pick models:", pd.Series(list(dframes.keys())), default=list(dframes.keys())
    )
    bodypart = st.selectbox("Pick a single bodypart:", pd.Series(bodypart_names))
    coordinate = st.radio("Coordinate:", pd.Series(["x", "y"]))
    cols = get_col_names(bodypart, coordinate, models)

    colors = px.colors.qualitative.Plotly

    fig_traces = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=[2, 1],
        vertical_spacing=0.05,
    )

    for c, col in enumerate(cols):
        fig_traces.add_trace(
            go.Scatter(
                name=col,
                x=np.arange(df_concat.shape[0]),
                y=df_concat[col],
                mode='lines',
                line=dict(color=colors[c]),
            ),
            row=1, col=1
        )

    for c, col in enumerate(cols):
        col_l = col.replace("_%s_" % coordinate, "_likelihood_")
        fig_traces.add_trace(
            go.Scatter(
                name=col_l,
                x=np.arange(df_concat.shape[0]),
                y=df_concat[col_l],
                mode='lines',
                line=dict(color=colors[c]),
                showlegend=False,
            ),
            row=2, col=1
        )

    fig_traces['layout']['yaxis']['title'] = "%s coordinate" % coordinate
    fig_traces['layout']['yaxis2']['title'] = "confidence"
    fig_traces.update_layout(
        width=800, height=600, title_text="Timeseries of %s (%s)" % (bodypart, coordinate))
    st.plotly_chart(fig_traces)

    # ---------------------------------------------------
    # plot temporal norms
    # ---------------------------------------------------

    st.header("Temporal loss diagnostics")
    x_label = "Model Name"
    y_label_tn = "Temporal Norm (pix)"

    big_df_temp_norm = compute_metric_per_dataset(
        dfs=dframes, metric="temporal_norm", bodypart_names=bodypart_names)
    # disp_temp_norms_head = st.checkbox("Display norms DataFrame")
    # if disp_temp_norms_head:
    #     st.write("Temporal norms dataframe:")
    #     st.write(big_df_temp_norm.head())

    # plot diagnostic averaged overall all bodyparts
    plot_type_tn = st.selectbox(
        "Pick a plot type:", ["box", "boxen", "bar", "violin", "strip"], key="plot_type_tn")
    plot_scale_tn = st.radio("Select y-axis scale", ["linear", "log"], key="plot_scale_tn")
    log_y_tn = False if plot_scale_tn == "linear" else True
    fig_cat_tn = make_seaborn_catplot(
        x="model_name", y="mean", data=big_df_temp_norm, log_y=log_y_tn, x_label=x_label,
        y_label=y_label_tn, title="Average over all keypoints", plot_type=plot_type_tn)
    st.pyplot(fig_cat_tn)

    # select bodypart to plot
    bodypart_temp_norm = st.selectbox(
        "Pick a single bodypart:", pd.Series([*bodypart_names, "mean"]), key="bodypart_temp_norm",
    )

    # show boxplot per bodypart
    fig_box_tn = make_plotly_catplot(
        x="model_name", y=bodypart_temp_norm, data=big_df_temp_norm, x_label=x_label,
        y_label=y_label_tn, title=bodypart_temp_norm, plot_type="box")
    st.plotly_chart(fig_box_tn)

    # show histogram per bodypart
    fig_hist_tn = make_plotly_catplot(
        x=bodypart_temp_norm, y=None, data=big_df_temp_norm, x_label=y_label_tn,
        y_label="Frame count", title=bodypart_temp_norm, plot_type="hist"
    )
    st.plotly_chart(fig_hist_tn)

    # # per bodypart
    # df_violin = big_df_norms.melt(id_vars="model_name")
    # fig = px.box(df_violin, x="model_name", y="value", color="variable", points=False)
    # st.plotly_chart(fig)

    # ---------------------------------------------------
    # plot multiview reprojection errors
    # ---------------------------------------------------

    uploaded_cfg: str = st.sidebar.file_uploader(
        "Select data config yaml (optional, for pca losses)", accept_multiple_files=False
    )
    if uploaded_cfg is not None:

        cfg = DictConfig(yaml.safe_load(uploaded_cfg))

        if cfg.data.get("mirrored_column_matches", None):

            st.header("PCA multiview loss diagnostics")
            y_label_pcamv = "Multiview PCA Reprojection Error (pix)"

            cfg_pcamv = cfg.copy()
            cfg_pcamv.model.losses_to_use = ["pca_multiview"]

            # compute pca loss
            pcamv_loss = build_pca_loss_object(cfg_pcamv)
            big_df_pcamv = compute_metric_per_dataset(
                dfs=dframes, metric="pca_mv", bodypart_names=bodypart_names, cfg=cfg_pcamv,
                pca_loss=pcamv_loss)

            # plot diagnostic averaged overall all bodyparts
            plot_type_pcamv = st.selectbox(
                "Pick a plot type:", ["box", "boxen", "bar", "violin", "strip"],
                key="plot_type_pcamv")
            plot_scale_pcamv = st.radio(
                "Select y-axis scale", ["linear", "log"], key="plot_scale_pcamv")
            log_y_pcamv = False if plot_scale_pcamv == "linear" else True
            fig_cat_pcamv = make_seaborn_catplot(
                x="model_name", y="mean", data=big_df_pcamv, log_y=log_y_pcamv,
                x_label=x_label, y_label=y_label_pcamv, title="Average over all keypoints",
                plot_type=plot_type_pcamv)
            st.pyplot(fig_cat_pcamv)

            # select bodypart to plot
            bodypart_pcamv = st.selectbox(
                "Pick a single bodypart:",
                pd.Series([*bodypart_names, "mean"]),
                key="bodypart_pcamv",
            )

            # show boxplot per bodypart
            fig_box_pcamv = make_plotly_catplot(
                x="model_name", y=bodypart_pcamv, data=big_df_pcamv, x_label=x_label,
                y_label=y_label_pcamv, title=bodypart_pcamv, plot_type="box")
            st.plotly_chart(fig_box_pcamv)

            # show histogram per bodypart
            fig_hist_pcamv = make_plotly_catplot(
                x=bodypart_pcamv, y=None, data=big_df_pcamv, x_label=y_label_pcamv,
                y_label="Frame count", title=bodypart_pcamv, plot_type="hist"
            )
            st.plotly_chart(fig_hist_pcamv)

        if cfg.data.get("columns_for_singleview_pca", None):

            st.header("PCA singleview loss diagnostics")
            y_label_pcasv = "Singleview PCA Reprojection Error (pix)"

            cfg_pcasv = cfg.copy()
            cfg_pcasv.model.losses_to_use = ["pca_singleview"]

            # compute pca loss
            pcasv_loss = build_pca_loss_object(cfg_pcasv)
            big_df_pcasv = compute_metric_per_dataset(
                dfs=dframes, metric="pca_sv", bodypart_names=bodypart_names, cfg=cfg_pcasv,
                pca_loss=pcasv_loss)

            # plot diagnostic averaged overall all bodyparts
            plot_type_pcasv = st.selectbox(
                "Pick a plot type:", ["box", "boxen", "bar", "violin", "strip"],
                key="plot_type_pcasv")
            plot_scale_pcasv = st.radio(
                "Select y-axis scale", ["linear", "log"], key="plot_scale_pcasv")
            log_y_pcasv = False if plot_scale_pcasv == "linear" else True
            fig_cat_pcasv = make_seaborn_catplot(
                x="model_name", y="mean", data=big_df_pcasv, log_y=log_y_pcasv,
                x_label=x_label, y_label=y_label_pcasv, title="Average over all keypoints",
                plot_type=plot_type_pcasv)
            st.pyplot(fig_cat_pcasv)

            # select bodypart to plot
            bodypart_pcasv = st.selectbox(
                "Pick a single bodypart:",
                pd.Series([*bodypart_names, "mean"]),
                key="bodypart_pcasv",
            )

            # show boxplot per bodypart
            fig_box_pcasv = make_plotly_catplot(
                x="model_name", y=bodypart_pcasv, data=big_df_pcasv, x_label=x_label,
                y_label=y_label_pcasv, title=bodypart_pcasv, plot_type="box")
            st.plotly_chart(fig_box_pcasv)

            # show histogram per bodypart
            fig_hist_pcasv = make_plotly_catplot(
                x=bodypart_pcasv, y=None, data=big_df_pcasv, x_label=y_label_pcasv,
                y_label="Frame count", title=bodypart_pcasv, plot_type="hist"
            )
            st.plotly_chart(fig_hist_pcasv)
