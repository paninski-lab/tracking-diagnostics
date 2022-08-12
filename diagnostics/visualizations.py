"""A collection of visualizations for various pose estimation performance metrics."""


def get_y_label(to_compute: str) -> str:
    if to_compute == 'rmse':
        return 'Pixel Error'
    if to_compute == 'temporal_norm':
        return 'Temporal norm (pix.)'
    elif to_compute == "pca_multiview":
        return "Multiview PCA \n reconstruction error (pix.)"
    elif to_compute == "pca_singleview":
        return "Low-dimensional PCA \n reconstruction error (pix.)"
    elif to_compute == "conf" or to_compute == "confidence":
        return "Confidence"
