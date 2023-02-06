import numpy as np
import os
import pandas as pd
from brainbox.behavior.dlc import get_pupil_diameter
from diagnostics.ensemble_kalman_filter import filtering_pass, smooth_backward, ensemble_median

def convert_lp_dlc(df_lp, keypoint_names, model_name='heatmap_tracker'):
    df_dlc = {} 
    for feat in keypoint_names: 
        for feat2 in ['x', 'y', 'likelihood']: 
            df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (model_name, feat, feat2)] 
    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc

def get_pupil_location(XYs):
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
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
    return center
    
def add_mean_to_array(pred_arr, keys, mean_x, mean_y):
    pred_arr_copy = pred_arr.copy()
    processed_arr_dict = {}
    for i, key in enumerate(keys):
        if 'x' in key:
            processed_arr_dict[key] = pred_arr_copy[:,i] + mean_x
        else:
            processed_arr_dict[key] = pred_arr_copy[:,i] + mean_y
    return processed_arr_dict


base_path = '/media/cat/cole/ibl-pupil_ensembling/'
video_name = '5285c561-80da-4563-8694-739da92e5dd0_left'

markers_list = []
diams = []
num_models = 10
keypoint_names = ['pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r']
for i in range(num_models):
    marker_path = os.path.join(base_path, f'{video_name}_rng={i}.csv')
    markers_tmp = pd.read_csv(marker_path, header=[0, 1, 2], index_col=0)
    if '.dlc' not in marker_path:
        markers_tmp = convert_lp_dlc(markers_tmp, keypoint_names, model_name='heatmap_tracker')
    diams_tmp = get_pupil_diameter(markers_tmp)
    markers_list.append(markers_tmp)
    diams.append(diams_tmp)

keys = ['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y', 'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']
#compute ensemble median
ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, keypoints_stack_dict = ensemble_median(markers_list, keys)

# # Kalman Filtering + Smoothing
# $z_t = (d_t, x_t, y_t)$
# $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
# $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

# ## Set parameters
#compute center of mass
pupil_locations = get_pupil_location(keypoints_mean_dict)
pupil_diameters = get_pupil_diameter(keypoints_mean_dict)
diameters = []
for i in range(num_models):
    keypoints_dict = keypoints_stack_dict[i]
    diameter = get_pupil_diameter(keypoints_dict)
    diameters.append(diameter)
    
mean_x_obs = np.mean(pupil_locations[:,0])
mean_y_obs = np.mean(pupil_locations[:,1])
x_t_obs, y_t_obs = pupil_locations[:,0] - mean_x_obs, pupil_locations[:,1] - mean_y_obs #make the mean zero

#latent variables (observed)
z_t_obs = np.vstack((pupil_diameters, x_t_obs, y_t_obs)) #latent variables - diameter, com_x, com_y

##### Set values for kalman filter #####
m0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0]) # initial state: mean
S0 =  np.asarray([[np.var(pupil_diameters), 0.0, 0.0], [0.0 , np.var(x_t_obs), 0.0], [0.0, 0.0 , np.var(y_t_obs)]]) # diagonal: var

A = np.asarray([[.999, 0, 0], [0, .999, 0], [0, 0, .999]]) #state-transition matrix, parameters hand-picked for smoothing purposes
Q = np.asarray([[np.var(pupil_diameters)*(1-(A[0,0]**2)), 0, 0], [0, np.var(x_t_obs)*(1-A[1,1]**2), 0], [0, 0, np.var(y_t_obs)*(1-(A[2,2]**2))]]) #state covariance matrix

#['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y', 'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y'] 
C = np.asarray([[0, 1, 0], [-.5, 0, 1], [0, 1, 0], [.5, 0, 1], [.5, 1, 0], [0, 0, 1], [-.5, 1, 0], [0, 0, 1]]) # Measurement function
R = np.eye(8) # placeholder diagonal matrix for ensemble variance
    
scaled_ensemble_preds = ensemble_preds.copy()
scaled_ensemble_stacks = ensemble_stacks.copy()
#subtract COM means from the ensemble predictions
for i in range(ensemble_preds.shape[1]):
    if i % 2 == 0:
        scaled_ensemble_preds[:, i] -= mean_x_obs
    else:
        scaled_ensemble_preds[:, i] -= mean_y_obs
#subtract COM means from all the predictions
for i in range(ensemble_preds.shape[1]):
    if i % 2 == 0:
        scaled_ensemble_stacks[:,:,i] -= mean_x_obs
    else:
        scaled_ensemble_stacks[:,:,i] -= mean_y_obs
y = scaled_ensemble_preds

## Perform filtering
#do filtering pass with time-varying ensemble variances
print("filtering...")
mf, Vf, S = filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
print("done filtering")

## Perform smoothing
# Do the smoothing step
print("smoothing...")
ms, Vs, _ = smooth_backward(y, mf, Vf, S, A, Q, C)
print("done smoothing")
# Smoothed posterior over y
y_m_smooth = np.dot(C, ms.T).T
y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

def make_dlc_pandas_index(keypoint_names):
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % 'ensemble-kalman'], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex

#make dataframes
pdindex = make_dlc_pandas_index(keypoint_names)

processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
key_pair_list = [['pupil_top_r_x', 'pupil_top_r_y'], ['pupil_right_r_x', 'pupil_right_r_y'], ['pupil_bottom_r_x', 'pupil_bottom_r_y'], ['pupil_left_r_x', 'pupil_left_r_y']]
pred_arr = []
for key_pair in key_pair_list:
    pred_arr.append(processed_arr_dict[key_pair[0]])
    pred_arr.append(processed_arr_dict[key_pair[1]])
    var = np.empty(processed_arr_dict[key_pair[0]].shape)
    var[:] = np.nan
    pred_arr.append(var)
pred_arr = np.asarray(pred_arr)
df = pd.DataFrame(pred_arr.T, columns=pdindex)
save_path = base_path + f'/kalman_smoothed_pupil_traces_{video_name}.csv'
print("saving latents to " + save_path)
df.to_csv(save_path)

pred_arr2 = []
pred_arr2.append(ms[:,0])
pred_arr2.append(ms[:,1] + mean_x_obs) #add back x mean of pupil location
pred_arr2.append(ms[:,2] + mean_y_obs) #add back y mean of pupil location
pred_arr2 = np.asarray(pred_arr2)
arrays = [['heatmap_tracker', 'heatmap_tracker', 'heatmap_tracker'], ['diameter', 'com_x', 'com_y']]
pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
df2 = pd.DataFrame(pred_arr2.T, columns=pd_index2)
save_path = base_path + f'/kalman_smoothed_latents_{video_name}.csv'
print("saving latents to " + save_path)
df2.to_csv(save_path)

