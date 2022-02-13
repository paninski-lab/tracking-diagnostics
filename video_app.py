"""
# My first app
I'd read the dataframes, split, get body part names, concatenate, then plot as I did
maybe left sidebar picks which datasets to read from a folder?
likelihood
TODO: add a csv file uploader, multifile
https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)

after you've uploaded 3 files, you give each model a name, manually
https://docs.streamlit.io/library/api-reference/widgets/st.text_input

display three such 
"""


from email.mime import base
from urllib.parse import _NetlocResultMixinBase
from grpc import dynamic_ssl_server_credentials
import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional


def strip_cols_append_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col + "_" + name for col in df.columns.values]
    return df


@st.cache
def concat_dfs(dframes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    counter = 0
    for model_name, dframe in dframes.items():
        if counter == 0:
            df_concat = dframe.copy()
            base_colnames = list(df_concat.columns.levels[0][1:])
            df_concat = strip_cols_append_name(df_concat, model_name)
        else:
            df = strip_cols_append_name(dframe.copy(), model_name)
            df_concat = pd.concat([df_concat, df], axis=1)
        counter += 1
    return df_concat, base_colnames


@st.cache
def compute_temporal_norms(
    df: pd.DataFrame, bodypart_names: List[str], model_name: str
):
    # compute the norm just for one dataframe
    df_norms = pd.DataFrame(columns=bodypart_names)
    diffs = df.diff(periods=1)  # not using .abs
    for col in bodypart_names:  # loop over bodyparts
        df_norms[col] = diffs[col][["x", "y"]].apply(
            np.linalg.norm, axis=1
        )  # norm of the difference for that bodypart
    df_norms["model_name"] = model_name
    df_norms["mean"] = df_norms[bodypart_names[:-1]].mean(axis=1)
    return df_norms


@st.cache
def compute_norms_per_dataset(
    dfs: Dict[str, pd.DataFrame], bodypart_names: List[str]
) -> pd.DataFrame:
    # cols.append("hparam")
    colnames = [*bodypart_names, "model_name"]
    concat_norm_df = pd.DataFrame(columns=colnames)
    for model_name, df in dfs.items():
        df_norm = compute_temporal_norms(df, bodypart_names, model_name)
        # compute mean across bps
        concat_norm_df = pd.concat([concat_norm_df, df_norm], axis=0)
    return concat_norm_df


def get_full_name(bodypart: str, coordinate: str, model: str) -> str:
    return "_".join([bodypart, coordinate, model])


def get_col_names(bodypart: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(bodypart, coordinate, model) for model in models]


st.title("Video Diagnostics")

st.sidebar.header("Data Settings")
uploaded_files: list = st.sidebar.file_uploader(
    "Choose one or more CSV files", accept_multiple_files=True
)

if len(uploaded_files) > 0:  # otherwise don't try to proceed
    # read dataframes into a dict with keys=filenames
    dframes = {}
    for uploaded_file in uploaded_files:
        dframes[uploaded_file.name] = pd.read_csv(uploaded_file, header=[1, 2])

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
    df_concat, bodypart_names = concat_dfs(dframes)

    st.header("Trace diagnostics")

    display_head = st.checkbox("Display trace DataFrame")
    if display_head:
        st.write("Concatenated Dataframe:")
        st.write(df_concat.head())


# the box select option should take body part, coordinate, and models out of checkbox
if len(uploaded_files) > 0:
    models = st.multiselect(
        "Pick models:", pd.Series(list(dframes.keys())), default=list(dframes.keys())
    )
    bodypart = st.selectbox("Pick a single bodypart:", pd.Series(bodypart_names))
    coordinate = st.radio("Coordinate:", pd.Series(["x", "y", "likelihood"]))
    # bodypart = 2
    cols = get_col_names(bodypart, coordinate, models)

    import plotly.express as px

    fig = px.line(
        df_concat,
        x=np.arange(df_concat.shape[0]),
        y=cols,
        labels={"x": "frame number", "value": coordinate},
        #               hover_data={"date": "|%B %d, %Y"},
        title="Timeseries of %s" % bodypart,
    )
    # add condition for this
    # files, data = load_data(NROWS)
    st.plotly_chart(fig)

    st.header("Temporal loss diagnostics")

    big_df_norms = compute_norms_per_dataset(dfs=dframes, bodypart_names=bodypart_names)
    disp_norms_head = st.checkbox("Display norms DataFrame")
    if disp_norms_head:
        st.write("Norms Dataframe:")
        st.write(big_df_norms.head())

    # show violin per bodypart
    models_norm = st.multiselect(
        "Pick models:",
        pd.Series(list(dframes.keys())),
        default=list(dframes.keys()),
        key="models_norm",
    )

    bodypart_norm = st.selectbox(
        "Pick a single bodypart:",
        pd.Series([*bodypart_names, "mean"]),
        key="models_norm",
    )

    single_bodypart_df = big_df_norms[[bodypart_norm, "model_name"]]

    # st.write(single_bodypart_df.head())

    fig_box = px.box(big_df_norms, x="model_name", y=bodypart_norm)
    fig_box.update_layout(
        yaxis_title="Temporal Norm", xaxis_title="Model Name", title=bodypart_norm,
    )
    st.plotly_chart(fig_box)

    fig_hist = px.histogram(
        big_df_norms,
        x=bodypart_norm,
        color="model_name",
        marginal="rug",
        barmode="overlay",
    )
    fig_hist.update_layout(
        yaxis_title="Frame count", xaxis_title="Temporal Norm", title=bodypart_norm,
    )
    st.plotly_chart(fig_hist)
    # df_violin = big_df_norms.melt(id_vars="model_name")

    # # per bodypart
    # fig = px.box(df_violin, x="model_name", y="value", color="variable", points=False)

    # st.plotly_chart(fig)

# compute norm, compute threshold crossings
# loop over original dataframes


# for i, vid in enumerate(video_names):
#     absolute_path_to_preds_file = os.path.join(video_dir, vid)
#     df_with_preds = pd.read_csv(absolute_path_to_preds_file, header=[1, 2])

#     splitted_vid_name = vid.split('_')[-1].split('.')
#     weight = '.'.join([splitted_vid_name[0], splitted_vid_name[1]]) # in front of temporal loss

#     if i == 0: # create big dataframe
#         col_names = df_with_preds.columns.levels[0][1:] # assuming all files have the same bp names
#         cols = list(col_names) # just bodypart names
#         cols.append("hparam") # adding a column called "hparam"
#         big_df = pd.DataFrame(columns = cols)

#     # compute the norm
#     df_norms = pd.DataFrame(columns = cols)
#     diffs = df_with_preds.diff(periods=1) # not using .abs
#     for col in col_names: # loop over bodyparts
#         df_norms[col] = diffs[col][["x", "y"]].apply(np.linalg.norm, axis=1) # norm of the difference for that bodypart
#         df_norms[col] = df_norms[col].mask(cond=df_norms[col]<eps, other=0.)
#     df_norms["hparam"] = weight # a scalar

#     big_df = pd.concat([big_df, df_norms]) # concat to big df
# assert(big_df.shape[0] == df_norms.shape[0]*len(video_names))


# # want: concat them all into a single
# @st.cache
# def concat_dataframes(
#     data_dict: Dict[str, pd.DataFrame],
#     files: List[str],
#     names: Optional[List[str]] = None,
# ) -> pd.DataFrame:

#     if names is None:
#         names = files  # do some splitting here

#     df_concat = data_dict[files[0]]
#     df_concat = strip_cols_append_name(df_concat, names[0])
#     for name, df in data_dict.items():
#         df = strip_cols_append_name(df, name)
# #         df_concat = pd.concat([df_concat, df], axis=1)

#     return df_concat


# # add a condition here whether to show a particular dataframe
# data[files[0]]

# df_concatal = concat_dataframes(data_dict=data, files=files)

# df_concatal

# df_concat = pd.read_csv(csv_paths[0], nrows=nrows, header=[1, 2])
# base_colnames = list(df_concat.columns.levels[0])[1:]  # before stripping
# df_concat = strip_cols_append_name(df_concat, model_names[0])
# for model_name, path in zip(model_names[1:], csv_paths[1:]):
#     df = pd.read_csv(path, nrows=nrows, header=[1, 2])
#     df = strip_cols_append_name(df, model_name)
#     df_concat = pd.concat([df_concat, df], axis=1)
# return df_concat, base_colnames
# pass


# @st.cache
# def load_data(
#     nrows: int, csv_paths: List[str], model_names: List[str]
# ) -> Tuple[pd.DataFrame, List[str]]:
#     for f in csv_paths:
#         assert f.endswith(".csv")
#     # loop that strips columns and concats models by column
#     # read single csv
#     df_concat = pd.read_csv(csv_paths[0], nrows=nrows, header=[1, 2])
#     base_colnames = list(df_concat.columns.levels[0])[1:]  # before stripping
#     df_concat = strip_cols_append_name(df_concat, model_names[0])
#     for model_name, path in zip(model_names[1:], csv_paths[1:]):
#         df = pd.read_csv(path, nrows=nrows, header=[1, 2])
#         df = strip_cols_append_name(df, model_name)
#         df_concat = pd.concat([df_concat, df], axis=1)
#     return df_concat, base_colnames


# TODO: add epsilon interface

# eps = 5.0
# raw_df_list = []
# norm_df_list = []
# for i, vid in enumerate(video_names):
#     absolute_path_to_preds_file = os.path.join(video_dir, vid)
#     df_with_preds = pd.read_csv(absolute_path_to_preds_file, header=[1, 2])

#     splitted_vid_name = vid.split('_')[-1].split('.')
#     weight = '.'.join([splitted_vid_name[0], splitted_vid_name[1]]) # in front of temporal loss

#     if i == 0: # create big dataframe
#         col_names = df_with_preds.columns.levels[0][1:] # assuming all files have the same bp names
#         cols = list(col_names) # just bodypart names
#         cols.append("hparam") # adding a column called "hparam"
#         big_df = pd.DataFrame(columns = cols)

#     # compute the norm
#     df_norms = pd.DataFrame(columns = cols)
#     diffs = df_with_preds.diff(periods=1) # not using .abs
#     for col in col_names: # loop over bodyparts
#         df_norms[col] = diffs[col][["x", "y"]].apply(np.linalg.norm, axis=1) # norm of the difference for that bodypart
#         df_norms[col] = df_norms[col].mask(cond=df_norms[col]<eps, other=0.)
#     df_norms["hparam"] = weight # a scalar

#     big_df = pd.concat([big_df, df_norms]) # concat to big df
# assert(big_df.shape[0] == df_norms.shape[0]*len(video_names))


# csv_paths = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR)]

# df_concat, bodypart_names = load_data(
#     nrows=NROWS, csv_paths=csv_paths, model_names=NAMES
# )
