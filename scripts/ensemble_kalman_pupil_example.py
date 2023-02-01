import glob
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import scipy
from brainbox.behavior.dlc import get_smooth_pupil_diameter, get_pupil_diameter
from diagnostics.ensemble_kalman_filter import filtering_pass, smooth_backward, ensemble_median, get_pupil_kalman_parameters
from lightning_pose.utils.predictions import PredictionHandler
import yaml

def convert_lp_dlc(df_lp, model_name='heatmap_tracker'):
    df_lp_convert = pd.DataFrame()
    # df['0'] = range(10000)
    df_lp_convert['pupil_top_r_x'] = df_lp[model_name][2:].astype('float64')
    df_lp_convert['pupil_top_r_y'] = df_lp[model_name+'.1'][2:].astype('float64')
    df_lp_convert['pupil_top_r_likelihood'] = df_lp[model_name+'.2'][2:].astype('float64')
    df_lp_convert['pupil_right_r_x'] = df_lp[model_name+'.3'][2:].astype('float64')
    df_lp_convert['pupil_right_r_y'] = df_lp[model_name+'.4'][2:].astype('float64')
    df_lp_convert['pupil_right_r_likelihood'] = df_lp[model_name+'.5'][2:].astype('float64')
    df_lp_convert['pupil_bottom_r_x'] = df_lp[model_name+'.6'][2:].astype('float64')
    df_lp_convert['pupil_bottom_r_y'] = df_lp[model_name+'.7'][2:].astype('float64')
    df_lp_convert['pupil_bottom_r_likelihood'] = df_lp[model_name+'.8'][2:].astype('float64')
    df_lp_convert['pupil_left_r_x'] = df_lp[model_name+'.9'][2:].astype('float64')
    df_lp_convert['pupil_left_r_y'] = df_lp[model_name+'.10'][2:].astype('float64')
    df_lp_convert['pupil_left_r_likelihood'] = df_lp[model_name+'.11'][2:].astype('float64')
    df_lp_convert.index += - 2 
    return df_lp_convert

def get_pupil_location(XYs, l_thresh=0.9, likelihoods=None, smooth=False):
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
    # direct diameters
    # t = XYs['pupil_top_r'][:, :2] / s
    # b = XYs['pupil_bottom_r'][:, :2] / s
    # l = XYs['pupil_left_r'][:, :2] / s
    # r = XYs['pupil_right_r'][:, :2] / s
    # print(np.vstack((XYs['pupil_top_r_x'].to_numpy(), XYs['pupil_top_r_y'].to_numpy())).T)
    t = np.vstack((XYs['pupil_top_r_x'], XYs['pupil_top_r_y'])).T / s
    b = np.vstack((XYs['pupil_bottom_r_x'], XYs['pupil_bottom_r_y'])).T / s
    l = np.vstack((XYs['pupil_left_r_x'], XYs['pupil_left_r_y'])).T / s
    r = np.vstack((XYs['pupil_right_r_x'], XYs['pupil_right_r_y'])).T / s
    center = np.zeros(t.shape)

    # ok if either top or bottom is nan in x-dir
    tmp_x1 = np.nanmedian(np.hstack([t[:, 0, None], b[:, 0, None]]), axis=1)
    # both left and right must be present in x-dir
    tmp_x2 = np.median(np.hstack([r[:, 0, None], l[:, 0, None]]), axis=1)
    center[:, 0] = np.nanmedian(np.hstack([tmp_x1[:, None], tmp_x2[:, None]]), axis=1)

    # both top and bottom must be present in y-dir
    tmp_y1 = np.median(np.hstack([t[:, 1, None], b[:, 1, None]]), axis=1)
    # ok if either left or right is nan in y-dir
    tmp_y2 = np.nanmedian(np.hstack([r[:, 1, None], l[:, 1, None]]), axis=1)
    center[:, 1] = np.nanmedian(np.hstack([tmp_y1[:, None], tmp_y2[:, None]]), axis=1)

    if smooth:
        return smooth_interpolate_signal_tv(center)
    else:
        return center
    
def process_pred_arr(pred_arr, keys):
    pred_arr_copy = pred_arr.copy()
    processed_arr_dict = {}
    for i, key in enumerate(keys):
        if 'x' in key:
            processed_arr_dict[key] = pred_arr_copy[:,i] + mean_x_obs
        else:
            processed_arr_dict[key] = pred_arr_copy[:,i] + mean_y_obs
    return processed_arr_dict


base_path = '/media/cat/cole/ibl-pupil_ensembling/'
video_name = '5285c561-80da-4563-8694-739da92e5dd0_left'

markers_list_baseline = []
diams_baseline = []
num_models = 10
for i in range(num_models):
    marker_path = os.path.join(base_path, f'{video_name}_rng={i}.csv')
    markers_tmp = pd.read_csv(marker_path)
    if '.dlc' not in marker_path:
        markers_tmp = convert_lp_dlc(markers_tmp, model_name='heatmap_tracker')
    diams_tmp = get_pupil_diameter(markers_tmp)
    markers_list_baseline.append(markers_tmp)
    diams_baseline.append(diams_tmp)

keys = ['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y', 'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']
#compute ensemble median
baseline_ensemble_preds, baseline_ensemble_vars, baseline_ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, keypoints_stack_dict = ensemble_median(markers_list_baseline, keys)

# # Kalman Filtering + Smoothing
# $z_t = (d_t, x_t, y_t)$
# $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
# $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

# ## Set parameters
#compute center of mass
pupil_locations = get_pupil_location(keypoints_mean_dict)
pupil_diameters = get_pupil_diameter(keypoints_mean_dict)
diameters = []
for i in range(10):
    keypoints_dict = keypoints_stack_dict[i]
    diameter = get_pupil_diameter(keypoints_dict)
    diameters.append(diameter)
m0, S0, A, Q, C, R, y, mean_x_obs, mean_y_obs = get_pupil_kalman_parameters(baseline_ensemble_preds, baseline_ensemble_stacks, pupil_locations, pupil_diameters)

scaled_baseline_ensemble_preds = baseline_ensemble_preds.copy()
scaled_baseline_ensemble_stacks = baseline_ensemble_stacks.copy()
#subtract COM means from the ensemble predictions
for i in range(baseline_ensemble_preds.shape[1]):
    if i % 2 == 0:
        scaled_baseline_ensemble_preds[:, i] -= mean_x_obs
    else:
        scaled_baseline_ensemble_preds[:, i] -= mean_y_obs
#subtract COM means from all the predictions
for i in range(baseline_ensemble_preds.shape[1]):
    if i % 2 == 0:
        scaled_baseline_ensemble_stacks[:,:,i] -= mean_x_obs
    else:
        scaled_baseline_ensemble_stacks[:,:,i] -= mean_y_obs


## Perform filtering
#do filtering pass with time-varying ensemble variances
print("filtering...")
mf, Vf, S = filtering_pass(y, m0, S0, C, R, A, Q, baseline_ensemble_vars)
print("done filtering")

## Perform smoothing
# Do the smoothing step
print("smoothing...")
ms, Vs, _ = smooth_backward(y, mf, Vf, S, A, Q, C)
print("done smoothing")

# Smoothed posterior over y
y_m_smooth = np.dot(C, ms.T).T
y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

#make dataframes
with open('/home/cole/lightning-pose/scripts/configs_ibl-pupil/config_ibl-pupil.yaml', 'r') as stream:
    try:
        cfg=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
cfg = OmegaConf.create(cfg)      

from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.scripts import (
    get_imgaug_transform, get_dataset, get_data_module, get_loss_factories,
)    

video_dir = '/media/cat/cole/ibl-pupil_ensembling/'
video_file = os.path.join(video_dir, f'%s.mp4' % video_name)
data_dir, _ = return_absolute_data_paths(data_cfg=cfg.data)
imgaug_transform = get_imgaug_transform(cfg=cfg)
dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
data_module.setup()
ph = PredictionHandler(cfg, data_module, video_file)
pdindex = ph.make_dlc_pandas_index()

processed_arr_dict = process_pred_arr(y_m_smooth, keys)

pred_arr = []
pred_arr.append(processed_arr_dict['pupil_top_r_x'])
pred_arr.append(processed_arr_dict['pupil_top_r_y'])
var = np.empty(processed_arr_dict['pupil_top_r_x'].shape)
var[:] = np.nan
pred_arr.append(var)

pred_arr.append(processed_arr_dict['pupil_right_r_x'])
pred_arr.append(processed_arr_dict['pupil_right_r_y'])
var = np.empty(processed_arr_dict['pupil_right_r_x'].shape)
var[:] = np.nan
pred_arr.append(var)

pred_arr.append(processed_arr_dict['pupil_bottom_r_x'])
pred_arr.append(processed_arr_dict['pupil_bottom_r_y'])
var = np.empty(processed_arr_dict['pupil_bottom_r_x'].shape)
var[:] = np.nan
pred_arr.append(var)

pred_arr.append(processed_arr_dict['pupil_left_r_x'])
pred_arr.append(processed_arr_dict['pupil_left_r_y'])
var = np.empty(processed_arr_dict['pupil_left_r_x'].shape)
var[:] = np.nan
pred_arr.append(var)

pred_arr = np.asarray(pred_arr)
pdindex = ph.make_dlc_pandas_index()
df = pd.DataFrame(pred_arr.T, columns=pdindex)
save_path = f'/media/cat/cole/kalman_smoothed_pupil_traces_{video_name}.csv'
print("saving latents to " + save_path)
df.to_csv(save_path)

pred_arr2 = []
pred_arr2.append(ms[:,0])
pred_arr2.append(ms[:,1] + mean_x_obs)
pred_arr2.append(ms[:,2] + mean_y_obs)
pred_arr2 = np.asarray(pred_arr2)

arrays = [['heatmap_tracker', 'heatmap_tracker', 'heatmap_tracker'], ['diameter', 'com_x', 'com_y']]
pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
df2 = pd.DataFrame(pred_arr2.T, columns=pd_index2)
save_path = f'/media/cat/cole/kalman_smoothed_latents_{video_name}.csv'
print("saving latents to " + save_path)
df2.to_csv(save_path)

