"""A collection of functions to assess pose estimation performance."""

import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
from typing import Optional

from lightning_pose.utils.pca import compute_pca_reprojection_error


def pca_reprojection_error_per_keypoint(
    pca_loss_object, keypoints_pred, device: Optional[str] = None
):
    """Compute error between data and its reprojection from a low-d representation."""
    if device is None:
        device = pca_loss_object.pca.device
    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(
            keypoints_pred, device=device, dtype=torch.float32
        )
    # TODO: check that the compute_loss function calls the right metric post hoc.
    # print(keypoints_pred.shape)
    keypoints_pred = pca_loss_object.pca._format_data(data_arr=keypoints_pred)
    # print(keypoints_pred.shape)
    elementwise_loss = pca_loss_object.compute_loss(predictions=keypoints_pred)
    # print(elementwise_loss.shape)
    return elementwise_loss.cpu().numpy()


def pca_reprojection_error(keypoints_pred, mean, kept_eigenvectors, device="cpu"):
    """Error between data and it's reprojection from a low-d representation.

    Args:
        keypoints_pred: np.ndarray or torch.Tensor, shape (samples, observation_dim)
        mean: np.ndarray or torch.Tensor, shape (observation_dim,)
        kept_eigenvectors: np.ndarray or torch.Tensor, shape
            (latent_dim, observation_dim)
        device: "cpu" | "cuda"

    Returns:
        np.ndarray, shape (samples, n_keypoints)

    """

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(
            keypoints_pred, device=device, dtype=torch.float32
        )
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    if not isinstance(kept_eigenvectors, torch.Tensor):
        kept_eigenvectors = torch.tensor(
            kept_eigenvectors, device=device, dtype=torch.float32
        )

    results = compute_pca_reprojection_error(
        clean_pca_arr=keypoints_pred,
        kept_eigenvectors=kept_eigenvectors,
        mean=mean,
    )

    return results.numpy()


def rmse(keypoints_true, keypoints_pred):
    """Root mean square error between true and predicted keypoints.

    Args:
        keypoints_true: np.ndarray, shape (samples, n_keypoints, 2)
        keypoints_pred: np.ndarray, shape (samples, n_keypoints, 2)

    Returns:
        np.ndarray, shape (samples, n_keypoints)

    """
    pixel_error = np.linalg.norm(keypoints_true - keypoints_pred, axis=2)
    return pixel_error
    # mse = np.square(keypoints_true - keypoints_pred)
    # return np.sqrt(0.5 * (mse[:, :, 0] + mse[:, :, 1]))


def average_precision(
        keypoints_true, keypoints_pred, scale, kappa, thresh=np.arange(0.5, 1.0, 0.05)):
    """Average Precision, as described in https://cocodataset.org/#keypoints-eval.

    Note this is similar to sklearns function sklearn.metrics.average_precision_score except it
    gives the user control over which thresholds to use for computation.

    Args:
        keypoints_true: np.ndarray
            shape (samples, n_keypoints, 2)
        keypoints_pred: np.ndarray
            shape (samples, n_keypoints, 2)
        scale: float
        kappa: float
        thresh: array-like
            array of OKS thresholds used for computing mAP

    Returns:
        np.ndarray, shape (n_keypoints,)

    """

    # flip thresh if necessary
    if not np.all(np.diff(thresh) < 0):
        thresh = np.flipud(thresh)

    # compute similarity based on this distance and other scaling factors
    similarity = OKS(keypoints_true, keypoints_pred, scale, kappa)

    if len(thresh) <= 1:
        raise ValueError("thresh must be an array with at least 2 elements.")

    precision = np.zeros((len(thresh), keypoints_true.shape[1]))
    recall = np.zeros_like(precision)
    for t, thresh_ in enumerate(thresh):
        for kp in range(keypoints_true.shape[1]):
            # define binary predictions by thresholding
            y_score = similarity[:, kp] >= thresh_
            # define ground truth as any keypoint that is labeled (so similarity is not nan)
            y_true = ~np.isnan(similarity[:, kp])
            # compute precision-recall
            precision[t, kp] = precision_score(y_true[y_true], y_score[y_true])
            recall[t, kp] = recall_score(y_true[y_true], y_score[y_true])

    ap = np.sum(np.diff(recall, axis=0) * precision[1:], axis=0)

    return ap


def OKS(keypoints_true, keypoints_pred, scale, kappa):
    """Object keypoint similarity metric; used to compute mAP.

    This metric is a similarity measure that is analagous to the intersection over union (IoU)
    metric commonly used in object detection.

    """
    # compute distance between true and predicted keypoints
    dist = np.linalg.norm(keypoints_true - keypoints_pred, axis=2)
    return np.exp(-np.square(dist) / (2.0 * np.square(scale * kappa)))


def unimodal_mse(heatmaps_pred, img_height, img_width, downsample_factor):
    """MSE between predicted heatmap and a unimodal heatmap constructed from its max.

    Args:
        heatmaps_pred: np.ndarray or torch.Tensor, shape
            (samples, n_keypoints, heatmap_height, heatmap_width)
        img_height: int
        img_width: int
        downsample_factor: int

    Returns:
        np.ndarray, shape (samples, n_keypoints)

    """

    from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
    from kornia.geometry.transform import pyrup
    from lightning_pose.data.utils import generate_heatmaps
    from lightning_pose.losses.losses import UnimodalLoss

    # initialize unimodal loss object
    uni_loss = UnimodalLoss(
        loss_name="unimodal_mse",
        original_image_height=img_height,
        original_image_width=img_width,
        downsampled_image_height=int(img_height // (2**downsample_factor)),
        downsampled_image_width=int(img_width // (2**downsample_factor)),
        data_module=None,
    )

    # check inputs
    if not isinstance(heatmaps_pred, torch.Tensor):
        heatmaps_pred = torch.tensor(
            heatmaps_pred, device=uni_loss.device, dtype=torch.float32
        )

    # construct unimodal heatmaps from predictions
    if isinstance(downsample_factor, int):
        for _ in range(downsample_factor):
            heatmaps_pred = pyrup(heatmaps_pred)
    softmaxes = spatial_softmax2d(
        heatmaps_pred, temperature=torch.tensor(100, device=uni_loss.device)
    )
    preds_pt = spatial_expectation2d(softmaxes, normalized_coordinates=False)
    heatmaps_ideal = generate_heatmaps(
        keypoints=preds_pt,
        height=uni_loss.original_image_height,
        width=uni_loss.original_image_width,
        output_shape=(
            uni_loss.downsampled_image_height,
            uni_loss.downsampled_image_width,
        ),
    )

    # compare unimodal heatmaps with predicted heatmaps
    results = uni_loss.compute_loss(
        targets=heatmaps_ideal,
        predictions=torch.tensor(heatmaps, device=uni_loss.device, dtype=torch.float32),
    ).numpy()

    return np.mean(results, axis=(2, 3))


def temporal_norm(keypoints_pred):
    """Norm of difference between keypoints on successive time bins.

    Args:
        keypoints_pred: np.ndarray or torch.Tensor, shape
            (samples, n_keypoints * 2) or (samples, n_keypoints, 2)

    Returns:
        np.ndarray, shape (samples - 1, n_keypoints)

    """

    from lightning_pose.losses.losses import TemporalLoss

    t_loss = TemporalLoss()

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(
            keypoints_pred, device=t_loss.device, dtype=torch.float32
        )

    if len(keypoints_pred.shape) != 2:
        keypoints_pred = keypoints_pred.reshape(keypoints_pred.shape[0], -1)

    return t_loss.compute_loss(keypoints_pred).numpy()


# --------------------------------------------------------------------------------------
# TODO
# --------------------------------------------------------------------------------------


def bad_likelihoods():
    """Determine whether likelihoods are below a given threshold for each body part."""
    pass


def skeleton_violations():
    """Compute distance between specified pairs of body parts on each frame."""
    pass


def pca_likelihood():
    """Evaluate the likelihood of body parts under the predictive covariance."""
    pass


def ensembling_error():
    """Compute error/variance across multiple network predictions for each keypoint."""
    pass
