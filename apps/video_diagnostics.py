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

Notes:
    - this file should only contain the streamlit logic for the user interface
    - data processing should come from (cached) functions imported from diagnsotics.reports
    - plots should come from (non-cached) functions imported from diagnostics.visualizations

"""

import argparse
import cv2
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
import streamlit as st
import yaml

from diagnostics.reports import build_metrics_df, concat_dfs, generate_report_video, get_col_names
from diagnostics.reports import ReportGenerator
from diagnostics.streamlit_utils import update_single_file, update_file_list
from diagnostics.visualizations import get_y_label
from diagnostics.visualizations import make_seaborn_catplot, make_plotly_catplot, plot_traces


@st.cache(allow_output_mutation=True)
def update_video_file(curr_file, new_file_list):
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

    st.title("Video Diagnostics")

    st.sidebar.header("Data Settings")
    uploaded_files_: list = st.sidebar.file_uploader(
        "Choose one or more CSV files", accept_multiple_files=True, type="csv",
    )
    # check to see if a prediction files were provided externally via cli arg
    uploaded_files, using_cli_preds = update_file_list(uploaded_files_, args.prediction_files)

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

        # upload video file
        # video_file_: str = st.sidebar.file_uploader(
        #     "Choose video file corresponding to predictions (optional, for labeled video)",
        #     accept_multiple_files=False,
        #     type="mp4",
        # )
        # TODO: cannot currently upload video from file explorer, doesn't return filepath
        # opencv VideoCapture cannot read a relative path or a BytesIO object
        video_file_ = None
        # check to see if a video file was provided externally via cli arg
        video_file = update_video_file(video_file_, args.video_file)
        if isinstance(video_file, Path):
            video_file = str(video_file)

        # ---------------------------------------------------
        # compute metrics
        # ---------------------------------------------------

        # concat dataframes, collapsing hierarchy and making df fatter.
        df_concat, keypoint_names = concat_dfs(dframes)
        df_metrics = build_metrics_df(
            dframes=dframes, keypoint_names=keypoint_names, is_video=True, cfg=cfg)
        metric_options = list(df_metrics.keys())

        # ---------------------------------------------------
        # plot diagnostics
        # ---------------------------------------------------

        # choose which metric to plot
        metric_to_plot = st.selectbox("Select a metric:", metric_options, key="metric")

        x_label = "Model Name"
        y_label = get_y_label(metric_to_plot)

        # plot diagnostic averaged overall all keypoints
        plot_type = st.selectbox("Select a plot type:", catplot_options, key="plot_type")
        plot_scale = st.radio("Select y-axis scale", scale_options, key="plot_scale")
        log_y = False if plot_scale == "linear" else True
        fig_cat = make_seaborn_catplot(
            x="model_name", y="mean", data=df_metrics[metric_to_plot], log_y=log_y, x_label=x_label,
            y_label=y_label, title="Average over all keypoints", plot_type=plot_type)
        st.pyplot(fig_cat)

        # select keypoint to plot
        keypoint_to_plot = st.selectbox(
            "Select a keypoint:", pd.Series([*keypoint_names, "mean"]), key="keypoint_to_plot",
        )
        # show boxplot per keypoint
        fig_box = make_plotly_catplot(
            x="model_name", y=keypoint_to_plot, data=df_metrics[metric_to_plot], x_label=x_label,
            y_label=y_label, title=keypoint_to_plot, plot_type="box")
        st.plotly_chart(fig_box)
        # show histogram per keypoint
        fig_hist = make_plotly_catplot(
            x=keypoint_to_plot, y=None, data=df_metrics[metric_to_plot], x_label=y_label,
            y_label="Frame count", title=keypoint_to_plot, plot_type="hist"
        )
        st.plotly_chart(fig_hist)

        # ---------------------------------------------------
        # plot traces
        # ---------------------------------------------------
        st.header("Trace diagnostics")

        models = st.multiselect(
            "Select models:", pd.Series(list(dframes.keys())), default=list(dframes.keys())
        )
        keypoint = st.selectbox("Select a keypoint:", pd.Series(keypoint_names))
        cols = get_col_names(keypoint, "x", models)
        fig_traces = plot_traces(df_metrics, df_concat, cols)
        st.plotly_chart(fig_traces)

        # ---------------------------------------------------
        # generate report
        # ---------------------------------------------------
        st.subheader("Generate diagnostic report")

        # select save directory
        st.text("current directory: %s" % os.getcwd())
        save_dir_ = st.text_input("Enter path of directory in which to save report")
        save_dir = ReportGenerator.generate_save_dir(base_save_dir=save_dir_, is_video=True)

        rpt_save_format = st.selectbox("Select figure format", ["pdf", "png"])

        rpt_n_frames = 500
        rpt_likelihood = 0.05
        rpt_framerate = 20
        rpt_single_vids = False
        if video_file is not None:
            rpt_n_frames = st.text_input("Number of frames in labeled video (<1000)", rpt_n_frames)
            rpt_likelihood = st.text_input("Likelihood threshold", rpt_likelihood)
            rpt_framerate = st.text_input("Labeled video framerate", rpt_framerate)
            rpt_single_vids = st.checkbox(
                "Output video for each individual bodypart", rpt_single_vids)

        st.markdown("""
            Click the `Generate Report` button below to automatically save out all plots. 
            Each available metric will be plotted. Options plot type and y-axis scale
            will be the same as those selected above. For each metric there will be one 
            overview plot that shows metrics for each individual keypoint, as well as another plot
            that shows the metric averaged across all keypoints.   
            
            **Note**: pca metrics will be computed and plotted when you upload a config yaml in the 
            left panel
        """)
        # * a labeled video will be created (using the same models whose traces are plotted
        # above) when you upload the video file in the left panel

        rpt_boxplot_type = plot_type
        rpt_boxplot_scale = plot_scale
        rpt_trace_models = models

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
            generate_report_video(
                df_traces=df_concat,
                df_metrics=df_metrics,
                keypoint_names=keypoint_names,
                save_dir=save_dir,
                format=rpt_save_format,
                box_kwargs={
                    "plot_type": rpt_boxplot_type,
                    "plot_scale": rpt_boxplot_scale,
                },
                trace_kwargs={
                    "model_names": rpt_trace_models,
                },
                savefig_kwargs=savefig_kwargs,
                video_kwargs={
                    "likelihood_thresh": float(rpt_likelihood),
                    "max_frames": int(rpt_n_frames),
                    "framerate": float(rpt_framerate),
                },
                video_file=video_file,
                make_video_per_keypoint=rpt_single_vids,
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
    parser.add_argument('--video_file', action='append', default=[])

    run()
