import numpy as np
import os
import pandas as pd

from diagnostics.ensemble_kalman_filter import ensemble_kalman_smoother_pupil


def convert_lp_dlc(df_lp, keypoint_names, model_name='heatmap_tracker'):
    df_dlc = {} 
    for feat in keypoint_names: 
        for feat2 in ['x', 'y', 'likelihood']: 
            df_dlc[f'{feat}_{feat2}'] = df_lp.loc[:, (model_name, feat, feat2)] 
    df_dlc = pd.DataFrame(df_dlc, index=df_lp.index)
    return df_dlc


base_path = '/media/cat/cole/ibl-pupil_ensembling/'
video_name = '5285c561-80da-4563-8694-739da92e5dd0_left'

markers_list = []
num_models = 10
keypoint_names = ['pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r']
for i in range(num_models):
    marker_path = os.path.join(base_path, f'{video_name}_rng={i}.csv')
    markers_tmp = pd.read_csv(marker_path, header=[0, 1, 2], index_col=0)
    if '.dlc' not in marker_path:
        markers_tmp = convert_lp_dlc(markers_tmp, keypoint_names, model_name='heatmap_tracker')
    markers_list.append(markers_tmp)

df_dicts = ensemble_kalman_smoother_pupil(markers_list, tracker_name='heatmap_tracker')

save_path = base_path + f'/kalman_smoothed_pupil_traces_{video_name}.csv'
print("saving smoothed predictions to " + save_path)
df_dicts['markers_df'].to_csv(save_path)

save_path = base_path + f'/kalman_smoothed_latents_{video_name}.csv'
print("saving latents to " + save_path)
df_dicts['latents_df'].to_csv(save_path)
