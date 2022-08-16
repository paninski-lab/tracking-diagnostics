"""A collection of visualizations for various pose estimation performance metrics."""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st


pix_error_key = "pixel error"
conf_error_key = "confidence"
temp_norm_error_key = "temporal norm"
pcamv_error_key = "pca multiview"
pcasv_error_key = "pca singleview"


# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
@st.cache
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


@st.cache
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
        return "Multiview PCA\nrecon error (pix.)"
    elif to_compute == "pca_singleview" or to_compute == "pca singleview":
        return "Low-dimensional PCA\nrecon error (pix.)"
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


def make_plotly_catplot(x, y, data, x_label, y_label, title, plot_type="box"):
    if plot_type == "box" or plot_type == "boxen":
        fig = px.box(data, x=x, y=y)
    elif plot_type == "violin":
        fig = px.violin(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
    elif plot_type == "strip":
        fig = px.strip(big_df_filtered, x="model_name", y=keypoint_to_plot, log_y=log_y)
    elif plot_type == "bar":
        fig = px.bar(big_df_filtered, x="model_name", y=keypoint_error, log_y=log_y)
    elif plot_type == "hist":
        fig = px.histogram(
            data, x=x, color="model_name", marginal="rug", barmode="overlay",
        )
    fig.update_layout(yaxis_title=y_label, xaxis_title=x_label, title=title)

    return fig


def plot_traces(df_metrics, df_traces, cols):

    # -------------------------------------------------------------
    # setup
    # -------------------------------------------------------------
    coordinate = "x"  # placeholder
    keypoint = cols[0].split("_%s_" % coordinate)[0]
    colors = px.colors.qualitative.Plotly

    rows = 3
    row_heights = [2, 2, 0.75]
    if temp_norm_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcamv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcasv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)

    fig_traces = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    yaxis_labels = {}
    row = 1

    # -------------------------------------------------------------
    # plot temporal norms, pcamv reproj errors, pcasv reproj errors
    # -------------------------------------------------------------
    for error_key in [temp_norm_error_key, pcamv_error_key, pcasv_error_key]:
        if error_key in df_metrics.keys():
            for c, col in enumerate(cols):
                # col = <keypoint>_<coord>_<model_name>.csv
                pieces = col.split("_%s_" % coordinate)
                if len(pieces) != 2:
                    # otherwise "_[x/y]_" appears in keypoint or model name :(
                    raise ValueError("invalid column name %s" % col)
                kp = pieces[0]
                model = pieces[1]
                fig_traces.add_trace(
                    go.Scatter(
                        name=col,
                        x=np.arange(df_traces.shape[0]),
                        y=df_metrics[error_key][kp][df_metrics[error_key].model_name == model],
                        mode='lines',
                        line=dict(color=colors[c]),
                        showlegend=False,
                    ),
                    row=row, col=1
                )
            if error_key == temp_norm_error_key:
                yaxis_labels['yaxis%i' % row] = "temporal<br>norm"
            elif error_key == pcamv_error_key:
                yaxis_labels['yaxis%i' % row] = "pca multi<br>error"
            elif error_key == pcasv_error_key:
                yaxis_labels['yaxis%i' % row] = "pca single<br>error"
            row += 1

    # -------------------------------------------------------------
    # plot traces
    # -------------------------------------------------------------
    for coord in ["x", "y"]:
        for c, col in enumerate(cols):
            pieces = col.split("_%s_" % coordinate)
            assert len(pieces) == 2  # otherwise "_[x/y]_" appears in keypoint or model name :(
            kp = pieces[0]
            model = pieces[1]
            new_col = col.replace("_%s_" % coordinate, "_%s_" % coord)
            fig_traces.add_trace(
                go.Scatter(
                    name=model,
                    x=np.arange(df_traces.shape[0]),
                    y=df_traces[new_col],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False if coord == "x" else True,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "%s coordinate" % coord
        row += 1

    # -------------------------------------------------------------
    # plot likelihoods
    # -------------------------------------------------------------
    for c, col in enumerate(cols):
        col_l = col.replace("_%s_" % coordinate, "_likelihood_")
        fig_traces.add_trace(
            go.Scatter(
                name=col_l,
                x=np.arange(df_traces.shape[0]),
                y=df_traces[col_l],
                mode='lines',
                line=dict(color=colors[c]),
                showlegend=False,
            ),
            row=row, col=1
        )
    yaxis_labels['yaxis%i' % row] = "confidence"
    row += 1

    # -------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------
    for k, v in yaxis_labels.items():
        fig_traces["layout"][k]["title"] = v
    fig_traces.update_layout(
        width=800, height=np.sum(row_heights) * 125,
        title_text="Timeseries of %s" % keypoint
    )

    return fig_traces
