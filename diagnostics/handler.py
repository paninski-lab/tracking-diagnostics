"""Model handler that loads models and computes metrics on videos/csv files."""

import os
import pandas as pd
import torch
import h5py
import numpy as np
from zmq import device

from diagnostics.io import find_model_versions
from diagnostics.metrics import (
    pca_reprojection_error_per_keypoint,
    rmse,
    temporal_norm,
    unimodal_mse,
)

from lightning_pose.utils.pca import (
    format_multiview_data_for_pca,
)


class ModelHandler(object):
    """Helper class for computing various metrics on pose estimates."""

    def __init__(self, base_dir, cfg, verbose=False):

        version = find_model_versions(base_dir, cfg, verbose=verbose)
        if len(version) == 0:
            raise FileNotFoundError("Did not find requested model in %s" % base_dir)
        self.model_dir = version[0]
        self.cfg = cfg
        self.pred_df = None
        self.last_computed_metric = None

    @staticmethod
    def resize_keypoints(cfg, keypoints_pred):
        """reshape to training dims for pca losses, which are optimized for these dims"""
        x_resize = cfg.data.image_resize_dims.width
        x_og = cfg.data.image_orig_dims.width
        keypoints_pred[:, :, 0] = keypoints_pred[:, :, 0] * (x_resize / x_og)
        # put y vals back in original pixel space
        y_resize = cfg.data.image_resize_dims.height
        y_og = cfg.data.image_orig_dims.height
        keypoints_pred[:, :, 1] = keypoints_pred[:, :, 1] * (y_resize / y_og)
        return keypoints_pred

    def compute_metric(self, metric, pred_file, video_file=None, **kwargs):
        """Compute a range of metrics on a provided prediction file.

        Args:
            metric:
                labeled data only:
                    "rmse": root mean square error between true and pred keypoints
                unlabeled data only:
                    "temporal_norm": keypoint-wise norm between successive time points
                either:
                    "pca_multiview": error between (2*num_views)-D preds and their 3D-PCA reprojection
                    "pca_singleview: error between preds and K-dim PCA reconstruction
                    "unimodal_mse":
            pred_file: absolute path of predictions csv file; if a relative path, the
                file is assumed to be in the model directory
            video_file: absolute path to video file; if pred_file does not exist, the
                model will be run on this video and stored at pred_file; an error is
                raised if pred_file does not exist and video_file is None
            **kwargs: the following are required for the indicated loss:
                "rmse":
                    keypoints_true: np.ndarray
                "temporal_norm":
                "pca_reproj":
                    pca_loss_obj: lightning_pose.utils.pca.KeypointPCA object
                "unimodal_mse":
                    heatmap_file: absolute path to heatmap h5 file; if not present, the
                        file is assumed to be in
                        `model_directory/heatmaps_and_images/heatmaps.h5`


        Returns:
            np.ndarray: desired metric computed for each frame

        """
        print("Metric: %s" % metric)
        # check paths
        if not os.path.isabs(pred_file):
            # assume file resides in model directory
            pred_file = os.path.join(self.model_dir, pred_file)

        # decide what to do if prediction file does not exist
        if not os.path.exists(pred_file):
            from lightning_pose.utils.io import ckpt_path_from_base_path

            if video_file is not None:
                assert os.path.isfile(video_file)
                # process video
                # - save pred keypoints at pred_file, which currently doesn't exist
                # - save pred heatmaps at heatmap_file if given in kwargs, else don't
                #   save
                from lightning_pose.utils.predictions import predict_single_video

                print("processing video at %s" % video_file)
                # handle paths
                saved_vid_preds_dir = os.path.dirname(pred_file)
                if not os.path.exists(saved_vid_preds_dir):
                    os.makedirs(saved_vid_preds_dir)
                if "heatmap_file" in kwargs.keys() and kwargs["heatmap_file"] is not None:
                    saved_heat_dir = os.path.dirname(kwargs["heatmap_file"])
                    if not os.path.exists(saved_heat_dir):
                        os.makedirs(saved_heat_dir)
                else:
                    kwargs["heatmap_file"] = None
                ckpt_file = ckpt_path_from_base_path(
                    self.model_dir, model_name=self.cfg.model.model_name
                )
                predict_single_video(
                    video_file=video_file,
                    ckpt_file=ckpt_file,
                    cfg_file=self.cfg,
                    preds_file=pred_file,
                    heatmap_file=kwargs["heatmap_file"],
                    sequence_length=self.cfg.eval.dali_parameters.sequence_length,
                )
            elif kwargs.get("datamodule", None) is not None:
                from lightning_pose.utils.predictions import predict_dataset

                print("processing labeled dataset")
                # handle paths
                saved_preds_dir = os.path.dirname(pred_file)
                if not os.path.exists(saved_preds_dir):
                    os.makedirs(saved_preds_dir)
                if "heatmap_file" in kwargs.keys():
                    saved_heat_dir = os.path.dirname(kwargs["heatmap_file"])
                    if not os.path.exists(saved_heat_dir):
                        os.makedirs(saved_heat_dir)
                else:
                    kwargs["heatmap_file"] = None
                ckpt_file = ckpt_path_from_base_path(
                    self.model_dir, model_name=self.cfg.model.model_name
                )
                predict_dataset(
                    cfg=self.cfg,
                    data_module=kwargs["datamodule"],
                    ckpt_file=ckpt_file,
                    preds_file=pred_file,
                    heatmap_file=kwargs["heatmap_file"],
                )
            else:
                raise FileNotFoundError("Did not find requested file at %s" % pred_file)

        # check input
        check_kwargs(kwargs, metric)

        # load predictions
        pred_df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
        if pred_df.keys()[-1][0] == "set":
            # these are predictions on labeled data; get rid of last column that
            # contains info about train/val/test set
            is_video = False
            tmp = pred_df.iloc[:, :-1].to_numpy().reshape(pred_df.shape[0], -1, 3)
        else:
            # these are predictions on video data
            is_video = True
            tmp = pred_df.to_numpy().reshape(pred_df.shape[0], -1, 3)
        keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)
        confidences = tmp[:, :, -1]  # shape (samples, n_keypoints)

        # compute metric
        if metric == "rmse":
            print("Computing RMSE...")
            if is_video:
                raise ValueError("cannot compute RMSE on unlabeled video data!")
            results = rmse(kwargs["keypoints_true"], keypoints_pred)

        elif metric == "pca_singleview" or metric == "pca_multiview":

            # resize back to smaller training dims.
            # TODO: be careful, that'll reshape all the keypoints going forward
            keypoints_pred = self.resize_keypoints(cfg, keypoints_pred=keypoints_pred)

            if metric == "pca_multiview":
                original_dims = keypoints_pred.shape
                mirrored_column_matches = kwargs["pca_loss_obj"].pca.mirrored_column_matches
                # adding a reshaping below since the loss class expects a single last dim with
                # num_keypoints*2
                results_raw = pca_reprojection_error_per_keypoint(
                    kwargs["pca_loss_obj"],
                    keypoints_pred=keypoints_pred.reshape(keypoints_pred.shape[0], -1))

                # original_dims = keypoints_pred.shape  # batch, num_total_keypoints, 2
                # print(original_dims)
                # mirrored_column_matches = kwargs["pca_obj"].mirrored_column_matches
                # keypoints_pred = torch.tensor(
                #     keypoints_pred,
                #     dtype=kwargs["pca_obj"].parameters["mean"].dtype,
                # )  # shape = (batch_size, num_keypoints, 2)
                # print("shape after making into a tensor:", keypoints_pred.shape)
                # keypoints_pred = format_multiview_data_for_pca(
                #     data_arr=keypoints_pred,
                #     mirrored_column_matches=mirrored_column_matches,
                # )  # shape = 2 * num_views X batch_size * num_used_keypoints
                # print("shape_after_formatting:", keypoints_pred.shape)

                results_raw = results_raw.reshape(
                    -1,
                    len(mirrored_column_matches[0]),
                    len(mirrored_column_matches),
                )  # batch X num_used_keypoints X num_views

                # next, put this back into a full keypoints pred arr
                results = np.nan * np.zeros(
                    (original_dims[0], original_dims[1] * 2)
                )  # removing the (x,y) coords, remaining with batch*num_total_keypoints
                for c, cols in enumerate(mirrored_column_matches):
                    results[:, cols] = results_raw[
                        :, :, c
                    ]  # just the columns belonging to view c
            if metric == "pca_singleview":
                # Dan commented this thing out in favor of the loss used for training.
                # results = pca_reprojection_error(
                #     keypoints_pred.reshape(
                #         keypoints_pred.shape[0], -1
                #     ),  # just keypoint preds
                #     mean=kwargs["pca_obj"].parameters["mean"],
                #     kept_eigenvectors=kwargs["pca_obj"].parameters["kept_eigenvectors"],
                #     device=kwargs["pca_obj"].device,
                # )
                results = pca_reprojection_error_per_keypoint(
                    kwargs["pca_loss_obj"],
                    keypoints_pred=keypoints_pred.reshape(keypoints_pred.shape[0], -1))

        elif metric == "unimodal_mse":
            if "heatmap_file" not in kwargs.keys() or kwargs["heatmap_file"] is None:
                # updated default path
                heatmap_file = os.path.join(self.model_dir, "heatmaps.h5")
                if not os.path.exists(heatmap_file):
                    # try old default path
                    heatmap_file = os.path.join(
                        self.model_dir, "heatmaps_and_images", "heatmaps.h5"
                    )
            else:
                heatmap_file = kwargs["heatmap_file"]
            with h5py.File(heatmap_file, "r") as f:
                heatmaps = f["heatmaps"][()]
                s = heatmaps.shape
            if s[0] != keypoints_pred.shape[0]:
                raise ValueError("Keypoints and heatmaps do not share batch dimension")
            results = unimodal_mse(
                heatmaps,
                img_height=self.cfg.data.image_resize_dims.height,
                img_width=self.cfg.data.image_resize_dims.width,
                downsample_factor=self.cfg.data.downsample_factor,
            )

        elif metric == "temporal_norm":
            if not is_video:
                raise ValueError("cannot compute temporal norm on labeled data!")
            results = temporal_norm(keypoints_pred)

        else:
            raise NotImplementedError

        self.pred_df = pred_df
        self.last_computed_metric = metric

        return results


def check_kwargs(kwargs, metric):
    """Ensure kwargs for metric computation have correct entries."""

    if metric == "rmse":
        req_kwargs = ["keypoints_true"]
    elif metric == "pca_multiview" or metric == "pca_singleview":
        req_kwargs = ["pca_loss_obj"]
    elif metric == "unimodal_mse":
        req_kwargs = []
    elif metric == "temporal_norm":
        req_kwargs = []
    else:
        raise NotImplementedError

    for req_kwarg in req_kwargs:
        if req_kwarg not in kwargs.keys():
            raise ValueError(
                "Must include %s in kwargs for %s computation" % (req_kwarg, metric)
            )
