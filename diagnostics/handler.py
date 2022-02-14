"""Model handler that loads models and computes metrics on videos/csv files."""

import os
import pandas as pd

from diagnostics.io import find_model_versions
from diagnostics.metrics import (
    pca_reprojection_error, rmse, temporal_norm, unimodal_mse
)


class ModelHandler(object):

    def __init__(self, base_dir, cfg, verbose=False):

        version = find_model_versions(base_dir, cfg, verbose=verbose)
        if len(version) == 0:
            raise FileNotFoundError("Did not find requested model in %s" % base_dir)
        self.model_dir = version[0]
        self.cfg = cfg
        self.pred_df = None
        self.last_computed_metric = None

    def compute_metric(self, metric, pred_file, video_file=None, **kwargs):
        """Compute a range of metrics on a provided prediction file.

        Args:
            metric:
                labeled data only:
                    "rmse": root mean square error between true and pred keypoints
                unlabeled data only:
                    "temporal_norm": keypoint-wise norm between successive time points
                either:
                    "pca_reproj": error between data and it's reprojection from a low-d
                        representation
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
                    pca_obj: lightning_pose.utils.pca.KeypointPCA object
                "unimodal_mse":
                    heatmap_file: absolute path to heatmap h5 file; if not present, the
                        file is assumed to be in
                        `model_directory/heatmaps_and_images/heatmaps.h5`


        Returns:
            np.ndarray: desired metric computed for each frame

        """

        # check paths
        if not os.path.isabs(pred_file):
            # assume file resides in model directory
            pred_file = os.path.join(self.model_dir, pred_file)

        # decide what to do if prediction file does not exist
        if not os.path.exists(pred_file):
            if video_file is not None:
                assert os.path.isfile(video_file)
                # process video
                # - save pred keypoints at pred_file, which currently doesn't exist
                # - save pred heatmaps at heatmap_file if given in kwargs, else don't
                #   save
                from lightning_pose.utils.io import ckpt_path_from_base_path
                from lightning_pose.utils.predictions import predict_single_video
                print('processing video at %s' % video_file)
                # handle paths
                saved_vid_preds_dir = os.path.dirname(pred_file)
                if not os.path.exists(saved_vid_preds_dir):
                    os.makedirs(saved_vid_preds_dir)
                if "heatmap_file" in kwargs.keys():
                    saved_heat_dir = os.path.dirname(kwargs["heatmap_file"])
                    if not os.path.exists(saved_heat_dir):
                        os.makedirs(saved_heat_dir)
                else:
                    kwargs["heatmap_file"] = None
                ckpt_file = ckpt_path_from_base_path(
                    self.model_dir, model_name=self.cfg.model.model_name)
                predict_single_video(
                    video_file=video_file,
                    ckpt_file=ckpt_file,
                    cfg_file=self.cfg,
                    preds_file=pred_file,
                    heatmap_file=kwargs["heatmap_file"],
                    sequence_length=self.cfg.eval.dali_parameters.sequence_length,
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
            if is_video:
                raise ValueError("cannot compute RMSE on unlabeled video data!")
            results = rmse(kwargs["keypoints_true"], keypoints_pred)

        elif metric == "pca_reproj":
            results = pca_reprojection_error(
                keypoints_pred.reshape(keypoints_pred.shape[0], -1),
                mean=kwargs["pca_obj"].parameters['mean'],
                kept_eigenvectors=kwargs["pca_obj"].parameters["kept_eigenvectors"],
                device=kwargs["pca_obj"].device,
            )

        elif metric == "unimodal_mse":
            if "heatmap_file" not in kwargs.keys() or kwargs["heatmap_file"] is None:
                # updated default path
                heatmap_file = os.path.join(self.model_dir, 'heatmaps.h5')
                if not os.path.exists(heatmap_file):
                    # try old default path
                    heatmap_file = os.path.join(
                        self.model_dir, 'heatmaps_and_images', 'heatmaps.h5')
            else:
                heatmap_file = kwargs["heatmap_file"]
            with h5py.File(heatmap_file, 'r') as f:
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
    elif metric == "pca_reproj":
        req_kwargs = ["pca_obj"]
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
