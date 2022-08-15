"""A collection of visualizations for various pose estimation performance metrics."""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
def get_df_box(df_orig, keypoint_names, model_names):
    df_boxes = []
    for keypoint in keypoint_names:
        for model_curr in model_names:
            tmp_dict = {
                "keypoint": keypoint,
                "metric": "Pixel error",
                "value": df_orig[df_orig.model_name == model_curr][keypoint],
                "model_name": model_curr,
            }
            df_boxes.append(pd.DataFrame(tmp_dict))
    return pd.concat(df_boxes)


def get_df_scatter(df_0, df_1, data_type, model_names, keypoint_names):
    df_scatters = []
    for keypoint in keypoint_names:
        df_scatters.append(pd.DataFrame({
            "img_file": df_0.img_file[df_0.set == data_type],
            "keypoint": keypoint,
            model_names[0]: df_0[keypoint][df_0.set == data_type],
            model_names[1]: df_1[keypoint][df_1.set == data_type],
        }))
    return pd.concat(df_scatters)


# ---------------------------------------------------
# PLOTTING
# ---------------------------------------------------

def get_y_label(to_compute: str) -> str:
    if to_compute == 'rmse' or to_compute == "pixel_error" or to_compute == "pixel error":
        return 'Pixel Error'
    if to_compute == 'temporal_norm' or to_compute == 'temporal norm':
        return 'Temporal norm (pix.)'
    elif to_compute == "pca_multiview" or to_compute == "pca multiview":
        return "Multiview PCA \n reconstruction error (pix.)"
    elif to_compute == "pca_singleview" or to_compute == "pca singleview":
        return "Low-dimensional PCA \n reconstruction error (pix.)"
    elif to_compute == "conf" or to_compute == "confidence":
        return "Confidence"


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
