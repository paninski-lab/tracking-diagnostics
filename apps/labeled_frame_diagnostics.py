"""Analyze predictions on labeled frames.

Users select an arbitrary number of csvs (one per model) from their file system

The app creates plots for:
- plot of a selected metric (e.g. pixel errors) for each model (bar/box/violin/etc)
- scatterplot of a selected metric between two models

to run from command line:
> streamlit run /path/to/labeled_frame_diagnostics.py

optionally, a ground truth labels file can be specified from the command line
(note the two sets of double dashes):
> streamlit run /path/to/labeled_frame_diagnostics.py -- --labels_csv=/path/to/file.csv

optionally, multiple prediction files can be specified from the command line; each must be
preceded by "--prediction_files":
> streamlit run /path/to/labeled_frame_diagnostics.py --
--prediction_files=/path/to/pred0.csv --prediction_files=/path/to/pred1.csv

optionally, names for each prediction file can be specified from the command line; each must be
preceded by "--model_names":
> streamlit run /path/to/labeled_frame_diagnostics.py --
--prediction_files=/path/to/pred0.csv --model_names=model0
--prediction_files=/path/to/pred1.csv --model_names=model1

optionally, a data config file can be specified from the command line
> streamlit run /path/to/labeled_frame_diagnostics.py -- --data_cfg=/path/to/cfg.yaml

Notes:
    - this file should only contain the streamlit logic for the user interface
    - data processing should come from (cached) functions imported from diagnsotics.reports
    - plots should come from (non-cached) functions imported from diagnostics.visualizations

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from omegaconf import DictConfig
import os
from typing import List, Dict, Tuple, Optional
import yaml

from diagnostics.reports import concat_dfs, build_metrics_df, get_df_box, get_df_scatter
from diagnostics.reports import ReportGenerator, generate_report_labeled
from diagnostics.streamlit_utils import update_single_file, update_file_list
from diagnostics.visualizations import make_seaborn_catplot, make_plotly_scatterplot, get_y_label

# TODO
# - refactor df making
# - show image on hover?


@st.cache(allow_output_mutation=True)
def update_cfg_file(curr_file, new_file_list):
    """Cannot use `update_single_file` for both or there will be cache collisons."""
    if curr_file is None and len(new_file_list) > 0:
        # pull file from cli args; wrap in Path so that it looks like an UploadedFile object
        # returned by streamlit's file_uploader
        ret_file = Path(new_file_list[0])
    else:
        ret_file = curr_file
    return ret_file


def increase_submits(n_submits=0):
    return n_submits + 1


st.session_state["n_submits"] = 0

catplot_options = ["boxen", "box", "bar", "violin", "strip"]
scale_options = ["linear", "log"]


def run():

    args = parser.parse_args()

    st.title("Labeled Frame Diagnostics")

    st.sidebar.header("Data Settings")

    # select ground truth label file from file system
    label_file_: str = st.sidebar.file_uploader(
        "Choose CSV file with labeled data", accept_multiple_files=False, type="csv",
    )
    # check to see if a label file was provided externally via cli arg
    label_file = update_single_file(label_file_, args.labels_csv)
    # if label_file is not None:
    #     dframe_gt = pd.read_csv(label_file, header=[1, 2], index_col=0)
    #     st.text(dframe_gt.head())

    prediction_files_: list = st.sidebar.file_uploader(
        "Choose one or more prediction CSV files", accept_multiple_files=True, type="csv",
    )
    # check to see if a prediction files were provided externally via cli arg
    prediction_files, using_cli_preds = update_file_list(prediction_files_, args.prediction_files)

    # col wrap when plotting results from all keypoints
    n_cols = 3

    if label_file is not None and len(prediction_files) > 0:  # otherwise don't try to proceed

        # ---------------------------------------------------
        # load data
        # ---------------------------------------------------

        # read dataframes into a dict with keys=filenames
        dframe_gt = pd.read_csv(label_file, header=[1, 2], index_col=0)
        if not isinstance(label_file, Path):
            label_file.seek(0)  # reset buffer after reading

        dframes = {}
        for p, prediction_file in enumerate(prediction_files):
            if using_cli_preds and len(args.model_names) > 0:
                # use provided names from cli if applicable
                filename = args.model_names[p]
            elif prediction_file.name in dframes.keys():
                # append new integer to duplicate filenames
                idx = 0
                new_name = "%s_0" % prediction_file.name
                while new_name in dframes.keys():
                    idx += 1
                    new_name = "%s_%i" % (prediction_file.name, idx)
                filename = new_name
            else:
                filename = prediction_file.name
            dframe = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
            if not isinstance(prediction_file, Path):
                prediction_file.seek(0)  # reset buffer after reading
            dframes[filename] = dframe
            data_types = dframe.iloc[:, -1].unique()

        # edit model names if desired, to simplify plotting
        st.sidebar.write("Model display names (editable)")
        new_names = []
        og_names = list(dframes.keys())
        for name in og_names:
            new_name = st.sidebar.text_input(label="", value=name)
            new_names.append(new_name)

        # change dframes key names to new ones
        for n_name, o_name in zip(new_names, og_names):
            dframes[n_name] = dframes.pop(o_name)

        # upload config file
        uploaded_cfg_: str = st.sidebar.file_uploader(
            "Select data config yaml (optional, for pca losses)", accept_multiple_files=False,
            type=["yaml", "yml"]
        )
        uploaded_cfg = update_cfg_file(uploaded_cfg_, args.data_cfg)
        if uploaded_cfg is not None:
            if isinstance(uploaded_cfg, Path):
                cfg = DictConfig(yaml.safe_load(open(uploaded_cfg)))
            else:
                cfg = DictConfig(yaml.safe_load(uploaded_cfg))
                uploaded_cfg.seek(0)  # reset buffer after reading
        else:
            cfg = None

        # ---------------------------------------------------
        # compute metrics
        # ---------------------------------------------------

        # concat dataframes, collapsing hierarchy and making df fatter.
        df_concat, keypoint_names = concat_dfs(dframes)
        df_metrics = build_metrics_df(
            dframes=dframes, keypoint_names=keypoint_names, is_video=False, cfg=cfg,
            dframe_gt=dframe_gt)
        metric_options = list(df_metrics.keys())

        # ---------------------------------------------------
        # user options
        # ---------------------------------------------------
        st.header("Select data to plot")

        # choose from individual keypoints, their mean, or all at once
        keypoint_to_plot = st.selectbox(
            "Select a keypoint:", ["mean", "ALL", *keypoint_names], key="keypoint")

        # choose which metric to plot
        metric_to_plot = st.selectbox("Select a metric:", metric_options, key="metric")
        y_label = get_y_label(metric_to_plot)

        # choose data split - train/val/test/unused
        data_type = st.selectbox("Select data partition:", data_types, key="data partition")

        # ---------------------------------------------------
        # plot metrics for all models
        # ---------------------------------------------------

        st.header("Compare multiple models")

        # enumerate plotting options
        plot_type = st.selectbox("Pick a plot type:", catplot_options)
        plot_scale = st.radio("Select y-axis scale", scale_options)

        # filter data
        df_metrics_filt = df_metrics[metric_to_plot][df_metrics[metric_to_plot].set == data_type]
        n_frames_per_dtype = df_metrics_filt.shape[0] // len(prediction_files)

        # plot data
        title = '%s (%i %s frames)' % (keypoint_to_plot, n_frames_per_dtype, data_type)

        log_y = False if plot_scale == "linear" else True

        if keypoint_to_plot == "ALL":

            df_box = get_df_box(df_metrics_filt, keypoint_names, new_names)
            sns.set_context("paper")
            fig_box = sns.catplot(
                x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
                kind=plot_type, data=df_box, height=2)
            fig_box.set_axis_labels("Model Name", y_label)
            fig_box.set_xticklabels(rotation=45, ha="right")
            fig_box.fig.subplots_adjust(top=0.94)
            fig_box.fig.suptitle("All keypoints (%i %s frames)" % (n_frames_per_dtype, data_type))
            st.pyplot(fig_box)

        else:

            fig_box = make_seaborn_catplot(
                x="model_name", y=keypoint_to_plot, data=df_metrics_filt, x_label="Model Name",
                y_label=y_label, title=title, log_y=log_y, plot_type=plot_type)
            st.pyplot(fig_box)

        # ---------------------------------------------------
        # scatterplots
        # ---------------------------------------------------

        st.header("Compare two models")
        model_0 = st.selectbox(
            "Model 0 (x-axis):", new_names, key="model_0")
        model_1 = st.selectbox(
            "Model 1 (y-axis):", [n for n in new_names if n != model_0], key="model_1")

        df_tmp0 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_0]
        df_tmp1 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_1]

        plot_scatter_scale = st.radio("Select axes scale", ["linear", "log"])

        if keypoint_to_plot == "ALL":

            df_scatter = get_df_scatter(
                df_tmp0, df_tmp1, data_type, [model_0, model_1], keypoint_names)
            fig_scatter = make_plotly_scatterplot(
                model_0=model_0, model_1=model_1, df=df_scatter,
                metric_name=y_label, title=title,
                axes_scale=plot_scatter_scale,
                facet_col="keypoint", n_cols=n_cols, hover_data=["img_file"],
                fig_height=300 * np.ceil(len(keypoint_names) / n_cols), fig_width=900,
            )

        else:

            df_scatter = pd.DataFrame({
                model_0: df_tmp0[keypoint_to_plot][df_tmp0.set == data_type],
                model_1: df_tmp1[keypoint_to_plot][df_tmp1.set == data_type],
                "img_file": df_tmp0.img_file[df_tmp0.set == data_type]
            })
            fig_scatter = make_plotly_scatterplot(
                model_0=model_0, model_1=model_1, df=df_scatter,
                metric_name=y_label, title=title,
                axes_scale=plot_scatter_scale,
                hover_data=["img_file"],
                fig_height=500, fig_width=500,
            )

        st.plotly_chart(fig_scatter)

        # ---------------------------------------------------
        # generate report
        # ---------------------------------------------------
        st.subheader("Generate diagnostic report")

        # select save directory
        st.text("current directory: %s" % os.getcwd())
        save_dir_ = st.text_input("Enter path of directory in which to save report")
        save_dir = ReportGenerator.generate_save_dir(base_save_dir=save_dir_, is_video=False)

        rpt_save_format = st.selectbox("Select figure format", ["pdf", "png"])

        st.markdown("""
            Click the `Generate Report` button below to automatically save out all plots. 
            Each available metric will be plotted. Options such as the data partition, plot type,
            etc. will be the same as those selected above. For each metric there will be one 
            overview plot that shows metrics for each individual keypoint, as well as another plot
            that shows the metric averaged across all keypoints.        
        
            **Note**: pca metrics will be computed and plotted when you upload a config yaml in the 
            left panel
        """)

        rpt_boxplot_type = plot_type
        rpt_boxplot_scale = plot_scale
        rpt_boxplot_dtype = data_type
        rpt_scatter_scale = plot_scatter_scale
        rpt_scatter_dtype = data_type
        rpt_model_0 = model_0
        rpt_model_1 = model_1

        # enumerate save options
        savefig_kwargs = {}

        disable_button = True if save_dir_ is None or save_dir_ == "" else False
        submit_report = st.button("Generate report", disabled=disable_button)
        if submit_report:
            st.warning("Generating report")
            if "n_submits" not in st.session_state:
                st.session_state["n_submits"] = 0
            else:
                st.session_state["n_submits"] = increase_submits(st.session_state["n_submits"])
            generate_report_labeled(
                df_metrics=df_metrics,
                keypoint_names=keypoint_names,
                model_names=new_names,
                save_dir=save_dir,
                format=rpt_save_format,
                box_kwargs={
                    "plot_type": rpt_boxplot_type,
                    "plot_scale": rpt_boxplot_scale,
                    "data_type": rpt_boxplot_dtype,
                },
                scatter_kwargs={
                    "plot_scale": rpt_scatter_scale,
                    "data_type": rpt_scatter_dtype,
                    "model_0": rpt_model_0,
                    "model_1": rpt_model_1,
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

    parser.add_argument('--labels_csv', action='append', default=[])
    parser.add_argument('--prediction_files', action='append', default=[])
    parser.add_argument('--model_names', action='append', default=[])
    parser.add_argument('--data_cfg', action='append', default=[])

    run()
