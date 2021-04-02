"""A collection of functions to assess pose estimation performance."""


def bad_likelihoods():
    """Determine whether likelihoods are below a given threshold for each body part."""
    pass


def jumps():
    """Compute temporal differences for each body part."""
    pass


def skeleton_violations():
    """Compute distance between specified pairs of body parts on each frame."""
    pass


def reprojection_errors():
    """Compute errors between original 2D points and reprojected 2D points from 3D estimates."""
    pass


def pca_likelihood():
    """Evaluate the likelihood of body parts under the predictive covariance."""
    pass


def ensembling_errors():
    """Compute the error (variance) across multiple network predictions for each body part."""
    pass
