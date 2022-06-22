"""Utility functions for streamlit apps."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from typing import List, Dict, Tuple, Optional

from lightning_pose.losses.losses import PCALoss
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    get_imgaug_transform, get_dataset, get_data_module, get_loss_factories,
)

from diagnostics.handler import ModelHandler
from diagnostics.metrics import rmse
from diagnostics.metrics import pca_reprojection_error_per_keypoint


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


def strip_cols_append_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col + "_" + name for col in df.columns.values]
    return df


def get_full_name(keypoint: str, coordinate: str, model: str) -> str:
    return "_".join([keypoint, coordinate, model])


def get_col_names(keypoint: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(keypoint, coordinate, model) for model in models]


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


# Plotly catplot
# if plot_type == "box" or plot_type == "boxen":
#     fig_box = px.box(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
# elif plot_type == "violin":
#     fig_box = px.violin(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
# elif plot_type == "strip":
#     fig_box = px.strip(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
# # elif plot_type == "bar":
# #     fig_box = px.bar(big_df_filtered, x="model_name", y=keypoint_error, log_y=log_y)
# else:
#     raise NotImplementedError
# fig_width = 500
# fig_height = 500
# fig_box.update_layout(
#     yaxis_title="Pixel Error", xaxis_title="Model Name", title=title,
#     width=fig_width, height=fig_height)
#
# st.plotly_chart(fig_box)
