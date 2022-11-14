"""Model handler that loads models and computes metrics on videos/csv files."""

import copy
import os
import pandas as pd
import torch
import h5py
import numpy as np

from lightning_pose.metrics import (
    pixel_error,
    temporal_norm,
    pca_singleview_reprojection_error,
    pca_multiview_reprojection_error,
)

from diagnostics.io import find_model_versions
from diagnostics.metrics import (
    OKS,
    average_precision,
    unimodal_mse,
)


class ModelHandler(object):
    """Helper class for computing various metrics on pose estimates."""

    def __init__(self, base_dir, cfg, verbose=False, keys_to_sweep=[], need_pred_csv=True):

        version = find_model_versions(
            base_dir, cfg, verbose=verbose, keys_to_sweep=keys_to_sweep,
            needs_pred_csv=need_pred_csv,
        )
        if len(version) == 0:
            raise FileNotFoundError("Did not find requested model in %s" % base_dir)
        self.model_dir = version[0]
        # make sure we don't turn on data augmentation
        cfg = copy.deepcopy(cfg)
        cfg.training.imgaug = "default"
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

    def compute_metric(self, metric, pred_file, video_file=None, confidence_thresh=0.0, **kwargs):
        """Compute a range of metrics on a provided prediction file.

        Args:
            metric:
                labeled data only:
                    "rmse"/"pixel_error": root mean square error between true and pred keypoints
                    "oks": object keypoint similarity
                    "average_precision": returns a value *per keypoint*, not per sample/keypoint
                unlabeled data only:
                    "temporal_norm": keypoint-wise norm between successive time points
                either:
                    "pca_multiview": error between (2*num_views)-D preds and their 3D-PCA reproj
                    "pca_singleview: error between preds and K-dim PCA reconstruction
                    "unimodal_mse": mse between predicted heatmap and its unimodal ideal
            pred_file: absolute path of predictions csv file; if a relative path, the
                file is assumed to be in the model directory
            video_file: absolute path to video file; if pred_file does not exist, the
                model will be run on this video and stored at pred_file; an error is
                raised if pred_file does not exist and video_file is None
            confidence_thresh : float
                set results to nan if model confidence is below this threshold
            **kwargs: the following are required for the indicated loss:
                "rmse" or "pixel_error":
                    keypoints_true: np.ndarray
                "oks":
                    scale: float
                    kappa: np.ndarray of shape (n_keypoints,)
                "temporal_norm":
                "pca_multiview"/"pca_singleview":
                    pca_loss_obj: lightning_pose.utils.pca.KeypointPCA object
                "unimodal_mse":
                    heatmap_file: absolute path to heatmap h5 file; if not present, the
                        file is assumed to be in
                        `model_directory/heatmaps_and_images/heatmaps.h5`
                other:
                    "datamodule": used to predict new dataset if pred_file does not exist



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
                )
            elif kwargs.get("datamodule", None) is not None:
                from lightning_pose.utils.predictions import predict_dataset

                print("processing labeled dataset")
                # handle paths
                saved_preds_dir = os.path.dirname(pred_file)
                if not os.path.exists(saved_preds_dir):
                    os.makedirs(saved_preds_dir)
                ckpt_file = ckpt_path_from_base_path(
                    self.model_dir, model_name=self.cfg.model.model_name
                )
                predict_dataset(
                    cfg=self.cfg,
                    data_module=kwargs["datamodule"],
                    ckpt_file=ckpt_file,
                    preds_file=pred_file,
                    gpu_id=self.cfg.training.gpu_id,
                )
            else:
                raise FileNotFoundError("Did not find requested file at %s" % pred_file)

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

        # check input
        check_kwargs(kwargs, metric, is_video)

        # compute metric
        if metric == "rmse" or metric == "pixel_error":
            results = pixel_error(kwargs["keypoints_true"], keypoints_pred)

        elif metric == "oks":
            results = OKS(
                kwargs["keypoints_true"], keypoints_pred, kwargs["scale"], kwargs["kappa"])

        elif metric == "average_precision":
            results = average_precision(
                kwargs["keypoints_true"], keypoints_pred, kwargs["scale"], kwargs["kappa"],
                kwargs.get("thresh", np.arange(0.5, 1.0, 0.05)))

        elif metric == "pca_singleview" or metric == "pca_multiview":
            results = pca_singleview_reprojection_error(
                keypoints_pred=keypoints_pred,
                pca=kwargs["pca_loss_obj"].pca,
                cfg=self.cfg,
            )

        elif metric == "pca_multiview":
            results = pca_multiview_reprojection_error(
                keypoints_pred=keypoints_pred,
                pca=kwargs["pca_loss_obj"].pca,
                cfg=self.cfg,
            )

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
            results = temporal_norm(keypoints_pred)

        else:
            raise NotImplementedError

        self.pred_df = pred_df
        self.last_computed_metric = metric

        if confidence_thresh > 0.0:
            assert results.shape == confidences.shape
            results[confidences < confidence_thresh] = np.nan

        return results, confidences


def check_kwargs(kwargs, metric, is_video):
    """Ensure kwargs for metric computation have correct entries."""

    if metric == "rmse":
        req_kwargs = ["keypoints_true"]
        req_video = False
        req_labels = True
    elif metric == "oks":
        req_kwargs = ["keypoints_true", "kappa", "scale"]
        req_video = False
        req_labels = True
    elif metric == "average_precision":
        req_kwargs = ["keypoints_true", "kappa", "scale"]
        req_video = False
        req_labels = True
    elif metric == "pca_multiview" or metric == "pca_singleview":
        req_kwargs = ["pca_loss_obj"]
        req_video = False
        req_labels = False
    elif metric == "unimodal_mse":
        req_kwargs = []
        req_video = False
        req_labels = False
    elif metric == "temporal_norm":
        req_kwargs = []
        req_video = True
        req_labels = False
    else:
        raise NotImplementedError

    if req_labels and is_video:
        raise ValueError("cannot compute %s on unlabeled video data!" % metric)
    if req_video and not is_video:
        raise ValueError("cannot compute %s on labeled data!" % metric)

    for req_kwarg in req_kwargs:
        if req_kwarg not in kwargs.keys():
            raise ValueError("Must include %s in kwargs for %s computation" % (req_kwarg, metric))
