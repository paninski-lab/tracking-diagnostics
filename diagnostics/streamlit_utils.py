"""Utility functions for streamlit apps."""

import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from typing import List, Dict, Tuple


@st.cache(allow_output_mutation=True)
def update_single_file(curr_file: str, new_file_list: list):
    if curr_file is None and len(new_file_list) > 0:
        # pull file from cli args; wrap in Path so that it looks like an UploadedFile object
        # returned by streamlit's file_uploader
        ret_file = Path(new_file_list[0])
    else:
        ret_file = curr_file
    return ret_file


@st.cache(allow_output_mutation=True)
def update_file_list(curr_file_list: list, new_file_list: list):
    use_cli_preds = False
    if len(curr_file_list) == 0 and len(new_file_list) > 0:
        # pull label file from cli args; wrap in Path so that it looks like an UploadedFile object
        # returned by streamlit's file_uploader
        ret_files = []
        for file in new_file_list:
            ret_files.append(Path(file))
        use_cli_preds = True
    else:
        ret_files = curr_file_list
    return ret_files, use_cli_preds
