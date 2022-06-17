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
from diagnostics.streamlit import build_pcamv_loss_object


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

    fig_box = px.box(big_df_temp_norm, x="model_name", y=bodypart_temp_norm)
    fig_box.update_layout(
        yaxis_title="Temporal Norm (pix)", xaxis_title="Model Name", title=bodypart_temp_norm,
    )
    st.plotly_chart(fig_box)

    fig_hist = px.histogram(
        big_df_temp_norm,
        x=bodypart_temp_norm,
        color="model_name",
        marginal="rug",
        barmode="overlay",
    )
    fig_hist.update_layout(
        yaxis_title="Frame count", xaxis_title="Temporal Norm (pix)", title=bodypart_temp_norm,
    )
    st.plotly_chart(fig_hist)
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
    # TODO: check that mirrored_column_matches exists, otherwise don't compute
    if uploaded_cfg is not None:

        st.header("PCA multiview loss diagnostics")

        cfg = DictConfig(yaml.safe_load(uploaded_cfg))

        cfg_pcamv = cfg.copy()
        cfg_pcamv.model.losses_to_use = ["pca_multiview"]

        # compute pca loss
        pca_loss = build_pcamv_loss_object(cfg_pcamv)
        big_df_pcamv = compute_metric_per_dataset(
            dfs=dframes, metric="pca_mv", bodypart_names=bodypart_names, cfg=cfg_pcamv,
            pca_loss=pca_loss)

        # show violin per bodypart
        bodypart_pcamv = st.selectbox(
            "Pick a single bodypart:",
            pd.Series([*bodypart_names, "mean"]),
            key="bodypart_pcamv",
        )

        fig_box = px.box(big_df_pcamv, x="model_name", y=bodypart_pcamv)
        fig_box.update_layout(
            yaxis_title="Multiview PCA Reprojection Error (pix)", xaxis_title="Model Name",
            title=bodypart_pcamv,
        )
        st.plotly_chart(fig_box)

        fig_hist = px.histogram(
            big_df_pcamv,
            x=bodypart_pcamv,
            color="model_name",
            marginal="rug",
            barmode="overlay",
        )
        fig_hist.update_layout(
            yaxis_title="Frame count", xaxis_title="Multiview PCA Reprojection Error (pix)",
            title=bodypart_pcamv,
        )
        st.plotly_chart(fig_hist)

# compute norm, compute threshold crossings
# loop over original dataframes


# for i, vid in enumerate(video_names):
#     absolute_path_to_preds_file = os.path.join(video_dir, vid)
#     df_with_preds = pd.read_csv(absolute_path_to_preds_file, header=[1, 2])

#     splitted_vid_name = vid.split('_')[-1].split('.')
#     weight = '.'.join([splitted_vid_name[0], splitted_vid_name[1]]) # in front of temporal loss

#     if i == 0: # create big dataframe
#         col_names = df_with_preds.columns.levels[0][1:] # assuming all files have the same bp names
#         cols = list(col_names) # just bodypart names
#         cols.append("hparam") # adding a column called "hparam"
#         big_df = pd.DataFrame(columns = cols)

#     # compute the norm
#     df_norms = pd.DataFrame(columns = cols)
#     diffs = df_with_preds.diff(periods=1) # not using .abs
#     for col in col_names: # loop over bodyparts
#         df_norms[col] = diffs[col][["x", "y"]].apply(np.linalg.norm, axis=1) # norm of the difference for that bodypart
#         df_norms[col] = df_norms[col].mask(cond=df_norms[col]<eps, other=0.)
#     df_norms["hparam"] = weight # a scalar

#     big_df = pd.concat([big_df, df_norms]) # concat to big df
# assert(big_df.shape[0] == df_norms.shape[0]*len(video_names))


# # want: concat them all into a single
# @st.cache
# def concat_dataframes(
#     data_dict: Dict[str, pd.DataFrame],
#     files: List[str],
#     names: Optional[List[str]] = None,
# ) -> pd.DataFrame:

#     if names is None:
#         names = files  # do some splitting here

#     df_concat = data_dict[files[0]]
#     df_concat = strip_cols_append_name(df_concat, names[0])
#     for name, df in data_dict.items():
#         df = strip_cols_append_name(df, name)
# #         df_concat = pd.concat([df_concat, df], axis=1)

#     return df_concat


# # add a condition here whether to show a particular dataframe
# data[files[0]]

# df_concatal = concat_dataframes(data_dict=data, files=files)

# df_concatal

# df_concat = pd.read_csv(csv_paths[0], nrows=nrows, header=[1, 2])
# base_colnames = list(df_concat.columns.levels[0])[1:]  # before stripping
# df_concat = strip_cols_append_name(df_concat, model_names[0])
# for model_name, path in zip(model_names[1:], csv_paths[1:]):
#     df = pd.read_csv(path, nrows=nrows, header=[1, 2])
#     df = strip_cols_append_name(df, model_name)
#     df_concat = pd.concat([df_concat, df], axis=1)
# return df_concat, base_colnames
# pass


# @st.cache
# def load_data(
#     nrows: int, csv_paths: List[str], model_names: List[str]
# ) -> Tuple[pd.DataFrame, List[str]]:
#     for f in csv_paths:
#         assert f.endswith(".csv")
#     # loop that strips columns and concats models by column
#     # read single csv
#     df_concat = pd.read_csv(csv_paths[0], nrows=nrows, header=[1, 2])
#     base_colnames = list(df_concat.columns.levels[0])[1:]  # before stripping
#     df_concat = strip_cols_append_name(df_concat, model_names[0])
#     for model_name, path in zip(model_names[1:], csv_paths[1:]):
#         df = pd.read_csv(path, nrows=nrows, header=[1, 2])
#         df = strip_cols_append_name(df, model_name)
#         df_concat = pd.concat([df_concat, df], axis=1)
#     return df_concat, base_colnames


# TODO: add epsilon interface

# eps = 5.0
# raw_df_list = []
# norm_df_list = []
# for i, vid in enumerate(video_names):
#     absolute_path_to_preds_file = os.path.join(video_dir, vid)
#     df_with_preds = pd.read_csv(absolute_path_to_preds_file, header=[1, 2])

#     splitted_vid_name = vid.split('_')[-1].split('.')
#     weight = '.'.join([splitted_vid_name[0], splitted_vid_name[1]]) # in front of temporal loss

#     if i == 0: # create big dataframe
#         col_names = df_with_preds.columns.levels[0][1:] # assuming all files have the same bp names
#         cols = list(col_names) # just bodypart names
#         cols.append("hparam") # adding a column called "hparam"
#         big_df = pd.DataFrame(columns = cols)

#     # compute the norm
#     df_norms = pd.DataFrame(columns = cols)
#     diffs = df_with_preds.diff(periods=1) # not using .abs
#     for col in col_names: # loop over bodyparts
#         df_norms[col] = diffs[col][["x", "y"]].apply(np.linalg.norm, axis=1) # norm of the difference for that bodypart
#         df_norms[col] = df_norms[col].mask(cond=df_norms[col]<eps, other=0.)
#     df_norms["hparam"] = weight # a scalar

#     big_df = pd.concat([big_df, df_norms]) # concat to big df
# assert(big_df.shape[0] == df_norms.shape[0]*len(video_names))


# csv_paths = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR)]

# df_concat, bodypart_names = load_data(
#     nrows=NROWS, csv_paths=csv_paths, model_names=NAMES
# )
