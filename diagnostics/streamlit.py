"""Utility functions for streamlit apps."""

import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Optional

from lightning_pose.losses.losses import PCALoss

from diagnostics.metrics import rmse


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


def get_full_name(bodypart: str, coordinate: str, model: str) -> str:
    return "_".join([bodypart, coordinate, model])


def get_col_names(bodypart: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(bodypart, coordinate, model) for model in models]


@st.cache(hash_funcs={PCALoss: lambda _: None})  # streamlit doesn't know how to hash PCALoss
def compute_metric_per_dataset(
    dfs: Dict[str, pd.DataFrame], metric: str, bodypart_names: List[str], **kwargs
) -> pd.DataFrame:

    # colnames = [*bodypart_names, "model_name"]
    # if "labels" in kwargs:
    #     colnames += ["imag_file"]
    # concat_df = pd.DataFrame(columns=colnames)
    concat_dfs = []
    for model_name, df in dfs.items():
        if metric == "rmse":
            df_ = compute_pixel_error(df, bodypart_names, model_name, **kwargs)
        elif metric == "temporal_norm":
            df_ = compute_temporal_norms(df, bodypart_names, model_name, **kwargs)
        elif metric == "pca_mv":
            df_ = compute_pcamv_reprojection_error(df, bodypart_names, model_name, **kwargs)
        else:
            raise NotImplementedError("%s is not a supported metric" % metric)
        # concat_df = pd.concat([concat_df, df_.reset_index(inplace=True, drop=False)], axis=0)
        # concat_df = pd.concat([concat_df, df_], axis=0, ignore_index=False)
    # return concat_df
        concat_dfs.append(df_)
    return pd.concat(concat_dfs)


@st.cache
def compute_pixel_error(
    df: pd.DataFrame, bodypart_names: List[str], model_name: str, labels: pd.DataFrame,
) -> pd.DataFrame:

    keypoints_true = labels.to_numpy().reshape(labels.shape[0], -1, 2) # shape (samples, n_keypoints, 2)

    tmp = df.iloc[:, :-1].to_numpy().reshape(df.shape[0], -1, 3)  # remove "set" column
    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)

    set = df.iloc[:, -1].to_numpy()

    results = rmse(keypoints_true, keypoints_pred)

    # collect results
    df_ = pd.DataFrame(columns=bodypart_names)
    for c, col in enumerate(bodypart_names):  # loop over bodyparts
        df_[col] = results[:, c]
    df_["model_name"] = model_name
    df_["mean"] = df_[bodypart_names[:-1]].mean(axis=1)
    df_["set"] = set
    df_["img_file"] = labels.index

    return df_


@st.cache
def compute_temporal_norms(
    df: pd.DataFrame, bodypart_names: List[str], model_name: str, **kwargs
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
def build_pcamv_loss_object(cfg):
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
    data_module.setup()
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)
    pca_loss = loss_factories["unsupervised"].loss_instance_dict["pca_multiview"]
    return pca_loss
