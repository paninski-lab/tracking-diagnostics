"""Analyze predictions on labeled frames.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- barplot of pixel errors for each model

to run from command line:
> streamlit run /path/to/labeled_frame_diagnostics.py

"""

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


st.title("Labeled Frame Diagnostics")

st.sidebar.header("Data Settings")
label_file: list = st.sidebar.file_uploader(
    "Choose CSV file with labeled data", accept_multiple_files=False
)
prediction_files: list = st.sidebar.file_uploader(
    "Choose one or more prediction CSV files", accept_multiple_files=True
)

if label_file is not None and len(prediction_files) > 0:  # otherwise don't try to proceed

    # ---------------------------------------------------
    # load data
    # ---------------------------------------------------

    # read dataframes into a dict with keys=filenames
    dframe_gt = pd.read_csv(label_file, header=[1, 2], index_col=0)

    dframes = {}
    for prediction_file in prediction_files:
        dframes[prediction_file.name] = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
        dtypes = dframes[prediction_file.name].iloc[:, -1].unique()
        print(dtypes)
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
    df_concat, bodypart_names = concat_dfs(dframes)

    # ---------------------------------------------------
    # plot pixel errors
    # ---------------------------------------------------

    st.header("Pixel errors")

    # display_head = st.checkbox("Display predictions DataFrame")
    # if display_head:
    #     st.write("Concatenated Dataframe:")
    #     st.write(df_concat.head())

    big_df_pix_error = compute_metric_per_dataset(
        dfs=dframes, metric='rmse', bodypart_names=bodypart_names, labels=dframe_gt)

    dtype = st.radio("Select data partition", dtypes)
    big_df_filtered = big_df_pix_error[big_df_pix_error.set == dtype]
    n_frames_per_dtype = big_df_filtered.shape[0] // len(prediction_files)

    bodypart_error = st.selectbox(
        "Pick a single bodypart:",
        pd.Series([*bodypart_names, "mean"]),
        key="bodypart_pix_error",
    )

    title = '%s (%i %s frames)' % (bodypart_error, n_frames_per_dtype, dtype)
    fig_box = px.box(big_df_filtered, x="model_name", y=bodypart_error)
    fig_box.update_layout(
        yaxis_title="Pixel Error", xaxis_title="Model Name", title=title,
    )
    st.plotly_chart(fig_box)
