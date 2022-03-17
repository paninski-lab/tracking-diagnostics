from omegaconf import OmegaConf
import streamlit as st
import hydra
import pandas as pd


# get config
dataset_name = "rick-configs"
# dataset_name = "ibl-pupil-2"
# dataset_name = "ibl-paw-2"
base_config_dir = "/home/jovyan/rick-configs"
base_save_dir = "/media/mattw/behavior/results/pose-estimation/"


@st.cache
def init_hydra(base_config_dir=base_config_dir):
    # only runs once
    hydra.initialize_config_dir(base_config_dir)


init_hydra()
cfg = hydra.compose(config_name="config")

main_keys = list(cfg.keys())
st.title("Model Comparison")
key_to_use = st.selectbox("Pick value:", pd.Series(main_keys))
keys_in_sub_cfg = cfg[key_to_use].keys()
# st.write(OmegaConf.to_object(cfg.keys()))
sub_key = st.selectbox("Pick key:", pd.Series(keys_in_sub_cfg))
def_val = cfg[key_to_use][sub_key]
st.write("default val:", def_val)
sub_value = st.text_input("Pick value, use comma delimiters if a list")

st.write(type(sub_value))
# TODO: this happens onl if list
st.write("new val/s:", sub_value)
str_list = sub_value.split(",")
st.write(str_list)
new_list = [int(sv) for sv in str_list]
st.write(new_list)
st.write(sub_value)

new_cfg = cfg.copy()
st.write(new_cfg)


# TODO: convert types if needed -- will be needed cause it's a string now
true_val = type(cfg[key_to_use][sub_key])  # convert types if needed


# type(sub_value)(true_val)
# st.write(type(sub_value))
# st.write(type(cfg[key_to_use][sub_key]))
# st.write(type(sub_value)(true_val))

# print a key to iterate on
