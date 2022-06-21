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
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from omegaconf import DictConfig
import os
from typing import List, Dict, Tuple, Optional
import yaml

from diagnostics.streamlit import get_col_names
from diagnostics.streamlit import concat_dfs
from diagnostics.streamlit import compute_metric_per_dataset
from diagnostics.streamlit import build_pca_loss_object


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

    display_head = st.checkbox("Display trace DataFrame")
    if display_head:
        st.write("Concatenated Dataframe:")
        st.write(df_concat.head())

    models = st.multiselect(
        "Pick models:", pd.Series(list(dframes.keys())), default=list(dframes.keys())
    )
    bodypart = st.selectbox("Pick a single bodypart:", pd.Series(bodypart_names))
    coordinate = st.radio("Coordinate:", pd.Series(["x", "y"]))
    # bodypart = 2
    cols = get_col_names(bodypart, coordinate, models)

    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=[2, 1],
        vertical_spacing=0.05,
    )

    for c, col in enumerate(cols):
        fig.add_trace(
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
        fig.add_trace(
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

    fig['layout']['yaxis']['title'] = "%s coordinate" % coordinate
    fig['layout']['yaxis2']['title'] = "confidence"
    fig.update_layout(width=800, height=600, title_text="Timeseries of %s" % bodypart)
    st.plotly_chart(fig)

    # ---------------------------------------------------
    # plot temporal norms
    # ---------------------------------------------------

    st.header("Temporal loss diagnostics")

    big_df_temp_norm = compute_metric_per_dataset(
        dfs=dframes, metric="temporal_norm", bodypart_names=bodypart_names)
    disp_temp_norms_head = st.checkbox("Display norms DataFrame")
    if disp_temp_norms_head:
        st.write("Temporal norms dataframe:")
        st.write(big_df_temp_norm.head())

    # show violin per bodypart
    # models_norm = st.multiselect(
    #     "Pick models:",
    #     pd.Series(list(dframes.keys())),
    #     default=list(dframes.keys()),
    #     key="models_norm",
    # )

    bodypart_temp_norm = st.selectbox(
        "Pick a single bodypart:",
        pd.Series([*bodypart_names, "mean"]),
        key="bodypart_temp_norm",
    )

    fig_pcamv_box = px.box(big_df_temp_norm, x="model_name", y=bodypart_temp_norm)
    fig_pcamv_box.update_layout(
        yaxis_title="Temporal Norm (pix)", xaxis_title="Model Name", title=bodypart_temp_norm,
    )
    st.plotly_chart(fig_pcamv_box)

    fig_pcamv_hist = px.histogram(
        big_df_temp_norm,
        x=bodypart_temp_norm,
        color="model_name",
        marginal="rug",
        barmode="overlay",
    )
    fig_pcamv_hist.update_layout(
        yaxis_title="Frame count", xaxis_title="Temporal Norm (pix)", title=bodypart_temp_norm,
    )
    st.plotly_chart(fig_pcamv_hist)
    # df_violin = big_df_norms.melt(id_vars="model_name")

    # # per bodypart
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

            cfg_pcamv = cfg.copy()
            cfg_pcamv.model.losses_to_use = ["pca_multiview"]

            # compute pca loss
            pcamv_loss = build_pca_loss_object(cfg_pcamv)
            big_df_pcamv = compute_metric_per_dataset(
                dfs=dframes, metric="pca_mv", bodypart_names=bodypart_names, cfg=cfg_pcamv,
                pca_loss=pcamv_loss)

            # show boxplot per bodypart
            bodypart_pcamv = st.selectbox(
                "Pick a single bodypart:",
                pd.Series([*bodypart_names, "mean"]),
                key="bodypart_pcamv",
            )

            fig_pcamv_box = px.box(big_df_pcamv, x="model_name", y=bodypart_pcamv)
            fig_pcamv_box.update_layout(
                yaxis_title="Multiview PCA Reprojection Error (pix)", xaxis_title="Model Name",
                title=bodypart_pcamv,
            )
            st.plotly_chart(fig_pcamv_box)

            # show histogram per bodypart
            fig_pcamv_hist = px.histogram(
                big_df_pcamv,
                x=bodypart_pcamv,
                color="model_name",
                marginal="rug",
                barmode="overlay",
            )
            fig_pcamv_hist.update_layout(
                yaxis_title="Frame count", xaxis_title="Multiview PCA Reprojection Error (pix)",
                title=bodypart_pcamv,
            )
            st.plotly_chart(fig_pcamv_hist)

        if cfg.data.get("columns_for_singleview_pca", None):

            st.header("PCA singleview loss diagnostics")

            cfg_pcasv = cfg.copy()
            cfg_pcasv.model.losses_to_use = ["pca_singleview"]

            # compute pca loss
            pcasv_loss = build_pca_loss_object(cfg_pcasv)
            big_df_pcasv = compute_metric_per_dataset(
                dfs=dframes, metric="pca_sv", bodypart_names=bodypart_names, cfg=cfg_pcasv,
                pca_loss=pcasv_loss)

            # show boxplot per bodypart
            bodypart_pcasv = st.selectbox(
                "Pick a single bodypart:",
                pd.Series([*bodypart_names, "mean"]),
                key="bodypart_pcasv",
            )

            fig_pcasv_box = px.box(big_df_pcasv, x="model_name", y=bodypart_pcasv)
            fig_pcasv_box.update_layout(
                yaxis_title="Singleview PCA Reprojection Error (pix)", xaxis_title="Model Name",
                title=bodypart_pcasv,
            )
            st.plotly_chart(fig_pcasv_box)

            # show histogram per bodypart
            fig_pcasv_hist = px.histogram(
                big_df_pcasv,
                x=bodypart_pcasv,
                color="model_name",
                marginal="rug",
                barmode="overlay",
            )
            fig_pcasv_hist.update_layout(
                yaxis_title="Frame count", xaxis_title="Singleview PCA Reprojection Error (pix)",
                title=bodypart_pcasv,
            )
            st.plotly_chart(fig_pcasv_hist)
