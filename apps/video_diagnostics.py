"""Analyze predictions on video data.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- time series/likelihoods of a selected keypoint (x or y coord) for each model
- boxplot/histogram of temporal norms for each model
- boxplot/histogram of multiview pca reprojection errors for each model

to run from command line:
> streamlit run /path/to/video_diagnostics.py

optionally, multiple prediction files can be specified from the command line; each must be
preceded by "--prediction_files":
> streamlit run /path/to/video_diagnostics.py --
--prediction_files=/path/to/pred0.csv --prediction_files=/path/to/pred1.csv

optionally, names for each prediction file can be specified from the command line; each must be
preceded by "--model_names":
> streamlit run /path/to/video_diagnostics.py --
--prediction_files=/path/to/pred0.csv --model_names=model0
--prediction_files=/path/to/pred1.csv --model_names=model1

optionally, a data config file can be specified from the command line
> streamlit run /path/to/video_diagnostics.py -- --data_cfg=/path/to/cfg.yaml

"""

import argparse
from datetime import datetime
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import seaborn as sns
import streamlit as st
from typing import List, Dict, Tuple, Optional
import yaml

from diagnostics.reports import generate_report_video
from diagnostics.streamlit import get_col_names
from diagnostics.streamlit import concat_dfs
from diagnostics.streamlit import compute_metric_per_dataset
from diagnostics.streamlit import build_pca_loss_object
from diagnostics.streamlit import update_single_file, update_file_list
from diagnostics.visualizations import make_seaborn_catplot, get_y_label, plot_traces
from diagnostics.visualizations import \
    conf_error_key, temp_norm_error_key, pcamv_error_key, pcasv_error_key


def make_plotly_catplot(x, y, data, x_label, y_label, title, plot_type="box"):

    if plot_type == "box":
        fig = px.box(data, x=x, y=y)
        fig.update_layout(yaxis_title=y_label, xaxis_title=x_label, title=title)
    elif plot_type == "hist":
        fig = px.histogram(
            data, x=x, color="model_name", marginal="rug", barmode="overlay",
        )
        fig.update_layout(yaxis_title=y_label, xaxis_title=x_label, title=title)

    return fig


def increase_submits(n_submits=0):
    return n_submits + 1


st.session_state["n_submits"] = 0

catplot_options = ["boxen", "box", "bar", "violin", "strip"]
scale_options = ["linear", "log"]


def run():

    args = parser.parse_args()

    st.title("Video Diagnostics")

    st.sidebar.header("Data Settings")
    uploaded_files_: list = st.sidebar.file_uploader(
        "Choose one or more CSV files", accept_multiple_files=True, type="csv",
    )
    # check to see if a prediction files were provided externally via cli arg
    uploaded_files, using_cli_preds = update_file_list(uploaded_files_, args.prediction_files)

    metric_options = []
    big_df = {}

    if len(uploaded_files) > 0:  # otherwise don't try to proceed

        # ---------------------------------------------------
        # load data
        # ---------------------------------------------------

        # read dataframes into a dict with keys=filenames
        dframes = {}
        for u, uploaded_file in enumerate(uploaded_files):
            if using_cli_preds and len(args.model_names) > 0:
                # use provided names from cli if applicable
                filename = args.model_names[u]
            elif uploaded_file.name in dframes.keys():
                # append new integer to duplicate filenames
                idx = 0
                new_name = "%s_0" % uploaded_file.name
                while new_name in dframes.keys():
                    idx += 1
                    new_name = "%s_%i" % (uploaded_file.name, idx)
                filename = new_name
            else:
                filename = uploaded_file.name
            dframes[filename] = pd.read_csv(uploaded_file, header=[1, 2], index_col=0)
            if not isinstance(uploaded_file, Path):
                uploaded_file.seek(0)  # reset buffer after reading

        # edit modelnames if desired, to simplify plotting
        st.sidebar.write("Model display names (editable)")
        new_names = []
        og_names = list(dframes.keys())
        for name in og_names:
            new_name = st.sidebar.text_input(label="", value=name)
            new_names.append(new_name)

        # change dframes key names to new ones
        for n_name, o_name in zip(new_names, og_names):
            dframes[n_name] = dframes.pop(o_name)

        # concat dataframes, collapsing hierarchy and making df fatter.
        df_concat, keypoint_names = concat_dfs(dframes)

        # ---------------------------------------------------
        # compute metrics
        # ---------------------------------------------------
        uploaded_cfg_: str = st.sidebar.file_uploader(
            "Select data config yaml (optional, for pca losses)", accept_multiple_files=False,
            type=["yaml", "yml"],
        )
        uploaded_cfg = update_single_file(uploaded_cfg_, args.data_cfg)
        if uploaded_cfg is not None:
            if isinstance(uploaded_cfg, Path):
                cfg = DictConfig(yaml.safe_load(open(uploaded_cfg)))
            else:
                cfg = DictConfig(yaml.safe_load(uploaded_cfg))
                uploaded_cfg.seek(0)  # reset buffer after reading
        else:
            cfg = None

        # confidence
        big_df[conf_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="confidence", keypoint_names=keypoint_names)
        metric_options += [conf_error_key]

        # temporal norm
        big_df[temp_norm_error_key] = compute_metric_per_dataset(
            dfs=dframes, metric="temporal_norm", keypoint_names=keypoint_names)
        metric_options += [temp_norm_error_key]

        # pca multiview
        if cfg is not None and cfg.data.get("mirrored_column_matches", None):
            cfg_pcamv = cfg.copy()
            cfg_pcamv.model.losses_to_use = ["pca_multiview"]
            pcamv_loss = build_pca_loss_object(cfg_pcamv)
            big_df[pcamv_error_key] = compute_metric_per_dataset(
                dfs=dframes, metric="pca_mv", keypoint_names=keypoint_names, cfg=cfg_pcamv,
                pca_loss=pcamv_loss)
            metric_options += [pcamv_error_key]

        # pca singleview
        if cfg is not None and cfg.data.get("columns_for_singleview_pca", None):
            cfg_pcasv = cfg.copy()
            cfg_pcasv.model.losses_to_use = ["pca_singleview"]
            pcasv_loss = build_pca_loss_object(cfg_pcasv)
            big_df[pcasv_error_key] = compute_metric_per_dataset(
                dfs=dframes, metric="pca_sv", keypoint_names=keypoint_names, cfg=cfg_pcasv,
                pca_loss=pcasv_loss)
            metric_options += [pcasv_error_key]

        # ---------------------------------------------------
        # plot diagnostics
        # ---------------------------------------------------

        # choose which metric to plot
        metric_to_plot = st.selectbox("Select a metric:", metric_options, key="metric")

        x_label = "Model Name"
        y_label = get_y_label(metric_to_plot)

        # plot diagnostic averaged overall all keypoints
        plot_type = st.selectbox(
            "Select a plot type:", ["boxen", "box", "bar", "violin", "strip"], key="plot_type")
        plot_scale = st.radio("Select y-axis scale", ["linear", "log"], key="plot_scale")
        log_y = False if plot_scale == "linear" else True
        fig_cat = make_seaborn_catplot(
            x="model_name", y="mean", data=big_df[metric_to_plot], log_y=log_y, x_label=x_label,
            y_label=y_label, title="Average over all keypoints", plot_type=plot_type)
        st.pyplot(fig_cat)

        # select keypoint to plot
        keypoint_to_plot = st.selectbox(
            "Select a keypoint:", pd.Series([*keypoint_names, "mean"]), key="keypoint_to_plot",
        )

        # show boxplot per keypoint
        fig_box = make_plotly_catplot(
            x="model_name", y=keypoint_to_plot, data=big_df[metric_to_plot], x_label=x_label,
            y_label=y_label, title=keypoint_to_plot, plot_type="box")
        st.plotly_chart(fig_box)

        # show histogram per keypoint
        fig_hist = make_plotly_catplot(
            x=keypoint_to_plot, y=None, data=big_df[metric_to_plot], x_label=y_label,
            y_label="Frame count", title=keypoint_to_plot, plot_type="hist"
        )
        st.plotly_chart(fig_hist)

        # # print(big_df[metric_to_plot].head())
        # df_tmp = big_df[metric_to_plot].melt(id_vars="model_name")
        # # print(df_tmp.head())
        # # print(df_tmp.columns)
        # fig_cat2 = sns.catplot(data=df_tmp, x="model_name", y="value", col="variable", col_wrap=3)
        # fig_cat2.set(yscale=plot_scale)
        # st.pyplot(fig_cat2)

        # ---------------------------------------------------
        # plot traces
        # ---------------------------------------------------

        st.header("Trace diagnostics")

        # display_head = st.checkbox("Display trace DataFrame")
        # if display_head:
        #     st.write("Concatenated Dataframe:")
        #     st.write(df_concat.head())

        models = st.multiselect(
            "Select models:", pd.Series(list(dframes.keys())), default=list(dframes.keys())
        )
        keypoint = st.selectbox("Select a keypoint:", pd.Series(keypoint_names))
        cols = get_col_names(keypoint, "x", models)
        fig_traces = plot_traces(big_df, df_concat, cols)
        st.plotly_chart(fig_traces)

        # ---------------------------------------------------
        # generate report
        # ---------------------------------------------------
        st.subheader("Generate diagnostic report")

        # select save directory
        run_date_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        # save_dir_default = os.path.join(os.getcwd(), run_date_time)
        st.text("current directory: %s" % os.getcwd())
        save_dir_ = st.text_input("Enter path of directory in which to save report")
        save_dir = os.path.join(save_dir_, "litpose-report-video_%s" % run_date_time)

        rpt_save_format = st.selectbox("Select figure format", ["pdf", "png"])

        st.markdown("""
            Click the `Generate Report` button below to automatically save out all plots. 
            Each available metric will be plotted. Options plot type and y-axis scale
            will be the same as those selected above. For each metric there will be one 
            overview plot that shows metrics for each individual keypoint, as well as another plot
            that shows the metric averaged across all keypoints.   
        """)

        rpt_boxplot_type = plot_type
        rpt_boxplot_scale = plot_scale
        rpt_trace_models = models

        # enumerate save options
        savefig_kwargs = {}

        submit_report = st.button("Generate report")
        if submit_report:
            if "n_submits" not in st.session_state:
                st.session_state["n_submits"] = 0
            else:
                st.session_state["n_submits"] = increase_submits(st.session_state["n_submits"])
            generate_report_video(
                df_traces=df_concat,
                df_metrics=big_df,
                keypoint_names=keypoint_names,
                model_names=new_names,
                save_dir=save_dir,
                format=rpt_save_format,
                box_kwargs={
                    "plot_type": rpt_boxplot_type,
                    "plot_scale": rpt_boxplot_scale,
                },
                trace_kwargs={
                    "models": rpt_trace_models,
                },
                savefig_kwargs=savefig_kwargs,
            )

        if st.session_state["n_submits"] > 0:
            msg = "Report directory located at<br>%s" % save_dir
            st.markdown(
                "<p style='font-family:sans-serif; color:Green;'>%s</p>" % msg,
                unsafe_allow_html=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_files', action='append', default=[])
    parser.add_argument('--model_names', action='append', default=[])
    parser.add_argument('--data_cfg', action='append', default=[])

    run()
