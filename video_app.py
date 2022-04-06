"""
# My first app
I'd read the dataframes, split, get body part names, concatenate, then plot as I did
maybe left sidebar picks which datasets to read from a folder?
likelihood
TODO: add a csv file uploader, multifile
https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)

after you've uploaded 3 files, you give each model a name, manually
https://docs.streamlit.io/library/api-reference/widgets/st.text_input

display three such

to run from command line:
> streamlit run /path/to/video_app.py

"""


from email.mime import base
from urllib.parse import _NetlocResultMixinBase
from grpc import dynamic_ssl_server_credentials
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


def strip_cols_append_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col + "_" + name for col in df.columns.values]
    return df


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
def compute_temporal_norms(
    df: pd.DataFrame, bodypart_names: List[str], model_name: str
) -> pd.DataFrame:
    # compute the norm just for one dataframe
    df_norms = pd.DataFrame(columns=bodypart_names)
    diffs = df.diff(periods=1)  # not using .abs
    for col in bodypart_names:  # loop over bodyparts
        df_norms[col] = diffs[col][["x", "y"]].apply(
            np.linalg.norm, axis=1
        )  # norm of the difference for that bodypart
    df_norms["model_name"] = model_name
    df_norms["mean"] = df_norms[bodypart_names[:-1]].mean(axis=1)
    return df_norms


@st.cache
def compute_norms_per_dataset(
    dfs: Dict[str, pd.DataFrame], bodypart_names: List[str]
) -> pd.DataFrame:

    colnames = [*bodypart_names, "model_name"]
    concat_norm_df = pd.DataFrame(columns=colnames)
    for model_name, df in dfs.items():
        df_norm = compute_temporal_norms(df, bodypart_names, model_name)
        concat_norm_df = pd.concat([concat_norm_df, df_norm], axis=0)
    return concat_norm_df


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def compute_pcamv_reprojection_error(
    df: pd.DataFrame, bodypart_names: List[str], model_name: str, cfg: dict, pca_loss: PCALoss,
) -> pd.DataFrame:

    # TODO: copied from diagnostics.handler.Handler::compute_metric; figure out a way to share
    tmp = df.to_numpy().reshape(df.shape[0], -1, 3)
    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)

    keypoints_pred = ModelHandler.resize_keypoints(cfg, keypoints_pred=keypoints_pred)

    original_dims = keypoints_pred.shape
    mirrored_column_matches = pca_loss.pca.mirrored_column_matches
    # adding a reshaping below since the loss class expects a single last dim with num_keypoints*2
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

    # collect results
    df_rpe = pd.DataFrame(columns=bodypart_names)
    for c, col in enumerate(bodypart_names):  # loop over bodyparts
        df_rpe[col] = results[:, c]
    df_rpe["model_name"] = model_name
    df_rpe["mean"] = df_rpe[bodypart_names[:-1]].mean(axis=1)
    return df_rpe


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def compute_pcamv_rpe_per_dataset(
    dfs: Dict[str, pd.DataFrame], bodypart_names: List[str], cfg: dict, pca_loss: PCALoss,
) -> pd.DataFrame:

    colnames = [*bodypart_names, "model_name"]
    concat_pcamv_df = pd.DataFrame(columns=colnames)
    for model_name, df in dfs.items():
        df_ = compute_pcamv_reprojection_error(df, bodypart_names, model_name, cfg, pca_loss)
        concat_pcamv_df = pd.concat([concat_pcamv_df, df_], axis=0)
    return concat_pcamv_df


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def build_pcamv_loss_object(cfg):
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
    data_module.setup()
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)
    pca_loss = loss_factories["unsupervised"].loss_instance_dict["pca_multiview"]
    return pca_loss


def get_full_name(bodypart: str, coordinate: str, model: str) -> str:
    return "_".join([bodypart, coordinate, model])


def get_col_names(bodypart: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(bodypart, coordinate, model) for model in models]


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
        dframes[uploaded_file.name] = pd.read_csv(uploaded_file, header=[1, 2], index_col=0)

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

    big_df_temp_norm = compute_norms_per_dataset(dfs=dframes, bodypart_names=bodypart_names)
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
        big_df_pcamv = compute_pcamv_rpe_per_dataset(
            dfs=dframes, bodypart_names=bodypart_names, cfg=cfg_pcamv, pca_loss=pca_loss)

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
