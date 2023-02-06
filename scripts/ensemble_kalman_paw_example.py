import numpy as np
import os
import pandas as pd
from brainbox.behavior.dlc import get_pupil_diameter
from diagnostics.ensemble_kalman_filter import filtering_pass, smooth_backward, ensemble_median
import scipy
from sklearn.decomposition import PCA

def convert_lp_dlc(df_lp, keypoint_names, model_name='heatmap_tracker'):
    df_dlc = {} 
    for feat in keypoint_names: 
        for feat2 in ['x', 'y', 'likelihood']: 
            df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (model_name, feat, feat2)] 
    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc

def remove_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        scaled_ensemble_stacks[k][:,0] = ensemble_stacks[k][:,0] - camera_means[0]
        scaled_ensemble_stacks[k][:,1] = ensemble_stacks[k][:,1] - camera_means[1]
        scaled_ensemble_stacks[k][:,2] = ensemble_stacks[k][:,2] - camera_means[2]
        scaled_ensemble_stacks[k][:,3] = ensemble_stacks[k][:,3] - camera_means[3]
    return scaled_ensemble_stacks

def add_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        scaled_ensemble_stacks[k][:,0] = ensemble_stacks[k][:,0] + camera_means[0]
        scaled_ensemble_stacks[k][:,1] = ensemble_stacks[k][:,1] + camera_means[1]
        scaled_ensemble_stacks[k][:,2] = ensemble_stacks[k][:,2] + camera_means[2]
        scaled_ensemble_stacks[k][:,3] = ensemble_stacks[k][:,3] + camera_means[3]
    return scaled_ensemble_stacks

def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps)
    return pca_.fit(S), pca_.explained_variance_ratio_

base_path = '/media/cat/cole/ibl-paw_ensembling/'
video_name = '032ffcdf-7692-40b3-b9ff-8def1fc18b2e'
paw = 'left'
s = 2 #smoothing param
quantile_keep_pca = 25

keypoint_names = ['paw_l', 'paw_r']
markers_list_left_cam = []
markers_list_right_cam = []
num_models = 10
for i in range(num_models):
    marker_path_left_cam = os.path.join(base_path, f'{video_name}.left.rng={i}.csv')
    markers_tmp_left_cam = pd.read_csv(marker_path_left_cam, header=[0, 1, 2], index_col=0)
    marker_path_right_cam = os.path.join(base_path, f'{video_name}.right.rng={i}.csv')
    markers_tmp_right_cam = pd.read_csv(marker_path_right_cam, header=[0, 1, 2], index_col=0)
    if '.dlc' not in marker_path_left_cam:
        markers_tmp_left_cam = convert_lp_dlc(markers_tmp_left_cam, keypoint_names)
    if '.dlc' not in marker_path_right_cam:
        markers_tmp_right_cam = convert_lp_dlc(markers_tmp_right_cam, keypoint_names)
    markers_list_left_cam.append(markers_tmp_left_cam)
    #switch right camera paws
    paw_l_x, paw_l_y, paw_l_likelihood = markers_tmp_right_cam['paw_l_x'].copy(), markers_tmp_right_cam['paw_l_y'].copy(), markers_tmp_right_cam['paw_l_likelihood'].copy()
    paw_r_x, paw_r_y, paw_r_likelihood = markers_tmp_right_cam['paw_r_x'].copy(), markers_tmp_right_cam['paw_r_y'].copy(), markers_tmp_right_cam['paw_r_likelihood'].copy()
    markers_tmp_right_cam['paw_l_x'], markers_tmp_right_cam['paw_l_y'], markers_tmp_right_cam['paw_l_likelihood'] = paw_r_x, paw_r_y, paw_r_likelihood
    markers_tmp_right_cam['paw_r_x'], markers_tmp_right_cam['paw_r_y'], markers_tmp_right_cam['paw_r_likelihood'] = paw_l_x, paw_l_y, paw_l_likelihood
    markers_list_right_cam.append(markers_tmp_right_cam)
time_stamps_left_cam = np.load(os.path.join(base_path, f'{video_name}.timestamps.left.npy'))
time_stamps_right_cam = np.load(os.path.join(base_path, f'{video_name}.timestamps.right.npy'))

markers_list_stacked_interp = []
markers_list_interp = [[],[]]

img_width = 128
for model_id in range(len(markers_list_left_cam)):
    bl_markers_curr = []
    left_markers_curr = []
    right_markers_curr = []
    bl_left_np = markers_list_left_cam[model_id].to_numpy()
    bl_right_np = markers_list_right_cam[model_id].to_numpy()
    bl_right_interp = []
    for i in range(bl_left_np.shape[1]):
        bl_right_interp.append(scipy.interpolate.interp1d(time_stamps_right_cam, bl_right_np[:, i]))
    for i, ts in enumerate(time_stamps_left_cam):
        if ts > time_stamps_right_cam[-1]:  
            break
        if ts < time_stamps_right_cam[0]:
            continue
        left_markers = np.array(bl_left_np[i, [0, 1, 3, 4]])
        left_markers_curr.append(left_markers)
        right_markers = np.array([bl_right_interp[j](ts) for j in [0, 1, 3, 4]])
        right_markers[0] = img_width - right_markers[0] #flip points to match left camera
        right_markers[2] = img_width - right_markers[2] #flip points to match left camera
        right_markers_curr.append(right_markers)
        bl_markers_curr.append(np.concatenate((left_markers[:2], right_markers[:2]))) #combine paw 1 predictions for both cameras
        bl_markers_curr.append(np.concatenate((left_markers[2:4], right_markers[2:4]))) #combine paw 2 predictions for both cameras
    markers_list_stacked_interp.append(bl_markers_curr)
    markers_list_interp[0].append(left_markers_curr)
    markers_list_interp[1].append(right_markers_curr)
markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
markers_list_interp = np.asarray(markers_list_interp)

keys = ['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y']
markers_list_left_cam = []
for k in range(len(markers_list_interp[0])):
    markers_left_cam = pd.DataFrame(markers_list_interp[0][k], columns=keys)
    markers_list_left_cam.append(markers_left_cam)
    
markers_list_right_cam = []
for k in range(len(markers_list_interp[1])):
    markers_right_cam = pd.DataFrame(markers_list_interp[1][k], columns=keys)
    markers_list_right_cam.append(markers_right_cam)
    
#compute ensemble median left camera
left_cam_ensemble_preds, left_cam_ensemble_vars, left_cam_ensemble_stacks, left_cam_keypoints_mean_dict, left_cam_keypoints_var_dict, left_cam_keypoints_stack_dict = ensemble_median(markers_list_left_cam, keys)

#compute ensemble median right camera
right_cam_ensemble_preds, right_cam_ensemble_vars, right_cam_ensemble_stacks, right_cam_keypoints_mean_dict, right_cam_keypoints_var_dict, right_cam_keypoints_stack_dict = ensemble_median(markers_list_right_cam, keys)

ensemble_stacked = np.median(markers_list_stacked_interp, 0)
ensemble_stacked_vars = np.var(markers_list_stacked_interp, 0)

#filter by low ensemble variances
hstacked_vars = np.hstack((left_cam_ensemble_vars, right_cam_ensemble_vars))
max_vars = np.max(hstacked_vars,1)
good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

good_left_cam_ensemble_preds = left_cam_ensemble_preds[good_frames]
good_right_cam_ensemble_preds = right_cam_ensemble_preds[good_frames]
good_left_cam_ensemble_vars = left_cam_ensemble_vars[good_frames]
good_right_cam_ensemble_vars = right_cam_ensemble_vars[good_frames]

good_stacked_ensemble_preds = np.zeros((good_frames.shape[0]*2, left_cam_ensemble_preds.shape[1]))
good_stacked_ensemble_vars = np.zeros((good_frames.shape[0]*2, left_cam_ensemble_vars.shape[1]))
i, j = 0, 0
while i < good_right_cam_ensemble_preds.shape[0]:
    good_stacked_ensemble_preds[j] = np.concatenate((good_left_cam_ensemble_preds[i][:2], good_right_cam_ensemble_preds[i][:2]))
    good_stacked_ensemble_vars[j] = np.concatenate((good_left_cam_ensemble_vars[i][:2], good_right_cam_ensemble_vars[i][:2]))
    j += 1
    good_stacked_ensemble_preds[j] = np.concatenate((good_left_cam_ensemble_preds[i][2:4], good_right_cam_ensemble_preds[i][2:4]))
    good_stacked_ensemble_vars[j] = np.concatenate((good_left_cam_ensemble_vars[i][2:4], good_right_cam_ensemble_vars[i][2:4]))
    i += 1
    j += 1
    
left_paw_ensemble_preds = np.zeros((left_cam_ensemble_preds.shape[0], left_cam_ensemble_preds.shape[1]))
right_paw_ensemble_preds = np.zeros((right_cam_ensemble_preds.shape[0], right_cam_ensemble_preds.shape[1]))
for i in range(len(left_cam_ensemble_preds)):
    left_paw_ensemble_preds[i] = np.concatenate((left_cam_ensemble_preds[i][:2], right_cam_ensemble_preds[i][:2]))
    right_paw_ensemble_preds[i] = np.concatenate((left_cam_ensemble_preds[i][2:4], right_cam_ensemble_preds[i][2:4]))
    
left_paw_ensemble_vars = np.zeros((left_cam_ensemble_vars.shape[0], left_cam_ensemble_vars.shape[1]))
right_paw_ensemble_vars = np.zeros((right_cam_ensemble_vars.shape[0], right_cam_ensemble_vars.shape[1]))
for i in range(len(left_cam_ensemble_vars)):
    left_paw_ensemble_vars[i] = np.concatenate((left_cam_ensemble_vars[i][:2], right_cam_ensemble_vars[i][:2]))
    right_paw_ensemble_vars[i] = np.concatenate((left_cam_ensemble_vars[i][2:4], right_cam_ensemble_vars[i][2:4]))
    
#get mean of each paw
mean_camera_l_x = good_stacked_ensemble_preds[:,0].mean()
mean_camera_l_y = good_stacked_ensemble_preds[:,1].mean()
mean_camera_r_x = good_stacked_ensemble_preds[:,2].mean()
mean_camera_r_y = good_stacked_ensemble_preds[:,3].mean()
means_camera = [mean_camera_l_x, mean_camera_l_y, mean_camera_r_x, mean_camera_r_y]

left_paw_ensemble_stacks = np.concatenate((left_cam_ensemble_stacks[:,:,:2], right_cam_ensemble_stacks[:,:,:2]),2)
scaled_left_paw_ensemble_stacks = remove_camera_means(left_paw_ensemble_stacks, means_camera)

right_paw_ensemble_stacks = np.concatenate((right_cam_ensemble_stacks[:,:,:2], right_cam_ensemble_stacks[:,:,:2]),2)
scaled_right_paw_ensemble_stacks = remove_camera_means(right_paw_ensemble_stacks, means_camera)

good_scaled_stacked_ensemble_preds = remove_camera_means(good_stacked_ensemble_preds[None,:,:], means_camera)[0]
ensemble_pca, ensemble_ex_var = pca(good_scaled_stacked_ensemble_preds, 3)

scaled_left_paw_ensemble_preds = remove_camera_means(left_paw_ensemble_preds[None,:,:], means_camera)[0]
ensemble_pcs_left_paw = ensemble_pca.transform(scaled_left_paw_ensemble_preds)
good_ensemble_pcs_left_paw = ensemble_pcs_left_paw[good_frames]

scaled_right_paw_ensemble_preds = remove_camera_means(right_paw_ensemble_preds[None,:,:], means_camera)[0]
ensemble_pcs_right_paw = ensemble_pca.transform(scaled_right_paw_ensemble_preds)
good_ensemble_pcs_right_paw = ensemble_pcs_right_paw[good_frames]

# # Kalman Filtering + Smoothing
# $z_t = (d_t, x_t, y_t)$
# $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
# $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

# ## Set parameters
if paw == 'left':
    save_keypoint_name = keypoint_names[0]
    good_ensemble_pcs = good_ensemble_pcs_left_paw
    ensemble_vars = left_paw_ensemble_vars
    y = scaled_left_paw_ensemble_preds
    ensemble_stacks = scaled_left_paw_ensemble_stacks
else:
    save_keypoint_name = keypoint_names[1]
    good_ensemble_pcs = good_ensemble_pcs_right_paw
    ensemble_vars = right_paw_ensemble_vars
    y = scaled_right_paw_ensemble_preds
    ensemble_stacks = scaled_right_paw_ensemble_stacks

#compute center of mass
#latent variables (observed)
good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

##### Set values for kalman filter #####
m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
S0 =  np.asarray([[np.var(good_z_t_obs[:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[:,2])]]) # diagonal: var

A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]
Q = s*np.cov(d_t.T)

#['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y', 'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y'] 
C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance

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

save_keypoint_names = ['l_cam_' + save_keypoint_name, 'r_cam_' + save_keypoint_name]
pdindex = make_dlc_pandas_index(save_keypoint_names)

scaled_y_m_smooth = add_camera_means(y_m_smooth[None,:,:], means_camera)[0]
pred_arr = []
for i in range(len(save_keypoint_names)):
    pred_arr.append(scaled_y_m_smooth.T[0 + 2*i])
    pred_arr.append(scaled_y_m_smooth.T[1 + 2*i])
    var = np.empty(scaled_y_m_smooth.T[0 + 2*i].shape)
    var[:] = np.nan
    pred_arr.append(var)
pred_arr = np.asarray(pred_arr)
df = pd.DataFrame(pred_arr.T, columns=pdindex)
save_path = base_path + f'/kalman_smoothed_paw_traces_{video_name}.csv'
print("saving latents to " + save_path)
df.to_csv(save_path)