"""Output a collection of plots that parallel those provided by the streamlit apps."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from diagnostics.streamlit import get_col_names
from diagnostics.visualizations import get_df_box, get_df_scatter
from diagnostics.visualizations import get_y_label, make_seaborn_catplot
from diagnostics.visualizations import plot_traces


def update_kwargs_dict_with_defaults(kwargs_new, kwargs_default):
    for key, val in kwargs_default.items():
        if key not in kwargs_new:
            kwargs_new[key] = val
    return kwargs_new


def generate_report_labeled(
        df,
        keypoint_names,
        model_names,
        save_dir,
        format="pdf",
        box_kwargs={},
        scatter_kwargs={},
        savefig_kwargs={}
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse inputs
    box_kwargs_default = {
        "plot_type": "boxen",
        "plot_scale": "log",
        "data_type": "test",
    }
    box_kwargs = update_kwargs_dict_with_defaults(box_kwargs, box_kwargs_default)

    scatter_kwargs_default = {
        "plot_scale": "log",
        "data_type": "test",
        "model_0": "model_0",
        "model_1": "model_1",
    }
    scatter_kwargs = update_kwargs_dict_with_defaults(scatter_kwargs, scatter_kwargs_default)

    metrics_to_plot = df.keys()

    # ---------------------------------------------------
    # plot metrics for all models (catplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        df_filtered = df[metric_to_plot][df[metric_to_plot].set == box_kwargs["data_type"]]
        n_frames_per_dtype = df_filtered.shape[0] // len(df_filtered.model_name.unique())

        y_label = get_y_label(metric_to_plot)

        # ---------------
        # plot ALL data
        # ---------------
        n_cols = 3
        df_box = get_df_box(df_filtered, keypoint_names, model_names)
        sns.set_context("paper")
        title = "All keypoints (%i %s frames)" % (n_frames_per_dtype, box_kwargs["data_type"])
        fig_box = sns.catplot(
            x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
            kind=box_kwargs["plot_type"], data=df_box, height=2)
        fig_box.set_axis_labels("Model Name", y_label)
        fig_box.set_xticklabels(rotation=45, ha="right")
        fig_box.set(yscale=box_kwargs["plot_scale"])
        fig_box.fig.subplots_adjust(top=0.94)
        fig_box.fig.suptitle(title)
        save_file = os.path.join(
            save_dir, "labeled-diagnostics_barplot-all_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

        # ---------------
        # plot mean data
        # ---------------
        keypoint_to_plot = "mean"
        title = 'Mean keypoints (%i %s frames)' % (n_frames_per_dtype, box_kwargs["data_type"])
        log_y = False if box_kwargs["plot_scale"] == "linear" else True
        make_seaborn_catplot(
            x="model_name", y=keypoint_to_plot, data=df_filtered, x_label="Model Name",
            y_label=y_label, title=title, log_y=log_y, plot_type=box_kwargs["plot_type"])
        save_file = os.path.join(
            save_dir, "labeled-diagnostics_barplot-mean_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

    # ---------------------------------------------------
    # plot metrics for a pair of models (scatterplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        model_0 = scatter_kwargs["model_0"]
        model_1 = scatter_kwargs["model_1"]

        df_tmp0 = df[metric_to_plot][df[metric_to_plot].model_name == model_0]
        df_tmp1 = df[metric_to_plot][df[metric_to_plot].model_name == model_1]

        y_label = get_y_label(metric_to_plot)
        xlabel_ = "%s<br>(%s)" % (y_label, model_0)
        ylabel_ = "%s<br>(%s)" % (y_label, model_1)

        log_scatter = False if scatter_kwargs["plot_scale"] == "linear" else True

        # ---------------
        # plot ALL data
        # ---------------
        n_cols = 3
        df_scatter = get_df_scatter(
            df_tmp0, df_tmp1, scatter_kwargs["data_type"], [model_0, model_1], keypoint_names)
        title = "All keypoints (%i %s frames)" % (n_frames_per_dtype, box_kwargs["data_type"])
        fig_scatter = px.scatter(
            df_scatter,
            x=model_0, y=model_1,
            facet_col="keypoint", facet_col_wrap=n_cols,
            log_x=log_scatter, log_y=log_scatter,
            opacity=0.5,
            # hover_data=['img_file'],
            # trendline="ols",
            title=title,
            labels={model_0: xlabel_, model_1: ylabel_},
        )
        fig_width = 900
        fig_height = 300 * np.ceil(len(keypoint_names) / n_cols)
        # clean up and save fig
        mn = np.min(df_scatter[[model_0, model_1]].min(skipna=True).to_numpy())
        mx = np.max(df_scatter[[model_0, model_1]].max(skipna=True).to_numpy())
        trace = go.Scatter(x=[mn, mx], y=[mn, mx], line_color="black", mode="lines")
        trace.update(legendgroup="trendline", showlegend=False)
        fig_scatter.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
        fig_scatter.update_layout(title=title, width=fig_width, height=fig_height)
        fig_scatter.update_traces(marker={'size': 5})
        save_file = os.path.join(
            save_dir, "labeled-diagnostics_scatterplot-all_%s.%s" % (metric_to_plot, format))
        fig_scatter.write_image(save_file)

        # ---------------
        # plot mean data
        # ---------------
        keypoint_to_plot = "mean"
        df_scatter = pd.DataFrame({
            model_0: df_tmp0[keypoint_to_plot][df_tmp0.set == scatter_kwargs["data_type"]],
            model_1: df_tmp1[keypoint_to_plot][df_tmp1.set == scatter_kwargs["data_type"]],
        })
        fig_scatter = px.scatter(
            df_scatter,
            x=model_0, y=model_1,
            log_x=log_scatter,
            log_y=log_scatter,
            opacity=0.5,
            # hover_data=['img_file'],
            # trendline="ols",
            title=title,
            labels={model_0: xlabel_, model_1: ylabel_},
        )
        fig_width = 500
        fig_height = 500
        # clean up and save fig
        mn = np.min(df_scatter[[model_0, model_1]].min(skipna=True).to_numpy())
        mx = np.max(df_scatter[[model_0, model_1]].max(skipna=True).to_numpy())
        trace = go.Scatter(x=[mn, mx], y=[mn, mx], line_color="black", mode="lines")
        trace.update(legendgroup="trendline", showlegend=False)
        fig_scatter.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
        fig_scatter.update_layout(title=title, width=fig_width, height=fig_height)
        fig_scatter.update_traces(marker={'size': 5})
        save_file = os.path.join(
            save_dir, "labeled-diagnostics_scatterplot-mean_%s.%s" % (metric_to_plot, format))
        fig_scatter.write_image(save_file)


def generate_report_video(
        df_traces,
        df_metrics,
        keypoint_names,
        model_names,
        save_dir,
        format="pdf",
        box_kwargs={},
        trace_kwargs={},
        savefig_kwargs={}
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # parse inputs
    box_kwargs_default = {
        "plot_type": "boxen",
        "plot_scale": "log",
    }
    box_kwargs = update_kwargs_dict_with_defaults(box_kwargs, box_kwargs_default)

    trace_kwargs_default = {
        "models": [],
    }
    trace_kwargs = update_kwargs_dict_with_defaults(trace_kwargs, trace_kwargs_default)

    metrics_to_plot = df_metrics.keys()

    # ---------------------------------------------------
    # plot metrics for all models (catplot)
    # ---------------------------------------------------
    for metric_to_plot in metrics_to_plot:

        x_label = "Model Name"
        y_label = get_y_label(metric_to_plot)

        # ---------------
        # plot ALL data
        # ---------------
        df_tmp = df_metrics[metric_to_plot].melt(id_vars="model_name")
        df_tmp = df_tmp.rename(columns={"variable": "keypoint"})
        fig_box = sns.catplot(
            data=df_tmp, x="model_name", y="value", col="keypoint", col_wrap=3,
            kind=box_kwargs["plot_type"]
        )
        fig_box.set(yscale=box_kwargs["plot_scale"])
        save_file = os.path.join(
            save_dir, "video-diagnostics_barplot-all_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

        # ---------------
        # plot mean data
        # ---------------
        log_y = False if box_kwargs["plot_scale"] == "linear" else True
        make_seaborn_catplot(
            x="model_name", y="mean", data=df_metrics[metric_to_plot], log_y=log_y,
            x_label=x_label, y_label=y_label, title="Average over all keypoints",
            plot_type=box_kwargs["plot_type"])
        save_file = os.path.join(
            save_dir, "video-diagnostics_barplot-mean_%s.%s" % (metric_to_plot, format))
        plt.savefig(save_file, dpi=300, format=format, **savefig_kwargs)

    # ---------------------------------------------------
    # plot traces
    # ---------------------------------------------------
    for keypoint in keypoint_names:

        cols = get_col_names(keypoint, "x", trace_kwargs["models"])
        fig_traces = plot_traces(df_metrics, df_traces, cols)
        save_file = os.path.join(
            save_dir, "video-diagnostics_traces-%s.%s" % (keypoint, format))
        fig_traces.write_image(save_file)
