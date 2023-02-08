"""Pipeline objects for analyzing ibl pupil and paw data in lightning pose paper."""

from brainbox.behavior.dlc import plt_window, insert_idx, WINDOW_LEN, SAMPLING, RESOLUTION
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import (
    bwm_query, load_good_units, load_trials_and_mask, merge_probes)
from brainwidemap.decoding.functions.decoding import fit_eid
from brainwidemap.decoding.functions.process_targets import load_behavior
from brainwidemap.decoding.functions.utils import get_save_path
from brainwidemap.decoding.settings_template import params
import cv2
import numpy as np
import one.alf.io as alfio
from one.api import ONE
import os
import pandas as pd
from pathlib import Path
from sklearn import linear_model as lm
import subprocess

from diagnostics.ensemble_kalman_filter import filtering_pass, smooth_backward, ensemble_median

# camera constants (from Michael Schartner)
FOCAL_LENGTH_MM = 16
SENSOR_SIZE = 12.7
IMG_WIDTH = 640
IMG_HEIGHT = 512

# downsampling factor used for paw inference
DS_FACTOR = 5

WINDOW_LAG = -0.4
conda_script = '/home/mattw/anaconda3/etc/profile.d/conda.sh'
conda_env = 'pose'


class Pipeline(object):

    def __init__(self, eid, one, view, base_dir=None, likelihood_thr=0.9):
        self.eid = eid
        self.one = one
        self.view = view

        # keep session loader on hand for easy loading
        self.sess_loader = SessionLoader(self.one, self.eid)
        self.sess_loader.load_trials()
        self.sess_loader.load_pose(views=[self.view], likelihood_thr=likelihood_thr)

        self.processed_video_name = None  # set by children classes

        # set paths
        self.paths = Paths()
        self.paths.alyx_session_path = one.eid2path(eid)
        if base_dir:
            self.paths.base_dir = base_dir
            self.paths.video_csv_dir = os.path.join(base_dir, 'video_preds')
            self.paths.kalman_save_dir = os.path.join(base_dir, 'kalman_outputs')
            self.paths.decoding_dir = os.path.join(base_dir, 'decoding')

    def pred_csv_file(self, rng_seed):
        return os.path.join(
            self.paths.video_csv_dir, f'{self.eid}.{self.view}.rng={rng_seed}.csv')

    @property
    def kalman_latents_file(self):
        return os.path.join(
            self.paths.kalman_save_dir, f'latents.kalman_smoothed.{self.eid}.{self.view}.csv')

    @property
    def kalman_markers_file(self):
        return os.path.join(
            self.paths.kalman_save_dir, f'markers.kalman_smoothed.{self.eid}.{self.view}.csv')

    def reencode_video(self, overwrite, video_file=None, **kwargs):
        """Re-encode video into yuv420p format needed for lightning pose."""
        # assume we'll process the standard video
        if video_file is None:
            video_in = os.path.join(
                self.paths.alyx_session_path, 'raw_video_data', self.processed_video_name)
        else:
            assert os.path.abspath(video_file)
            video_in = video_file
        video_out = video_in.replace('.mp4', '_reencode.mp4')
        if os.path.exists(video_out) and not overwrite:
            print(f'{video_out} already exists; skipping')
            return video_out
        call_str = \
            f'ffmpeg -i {video_in} -profile:v main -pix_fmt yuv420p -crf 17 -vsync 0 {video_out}'
        subprocess.run(['/bin/bash', '-c', call_str], check=True)
        return video_out

    def infer_video(
            self, model_dir, data_dir, pred_csv_file, video_file=None, gpu_id=0, overwrite=False):
        """Can eventually just run litpose code directly from this function; doing this now to
        avoid installing litpose in iblenv conda environment.
        """

        # skip if model has not finished training
        pred_file_og = os.path.join(model_dir, 'predictions.csv')
        if not os.path.exists(pred_file_og):
            raise FileNotFoundError(
                f'{pred_file_og} does not exist; has your model finished training?')

        if os.path.exists(pred_csv_file) and not overwrite:
            print(f'{pred_csv_file} already exists; skipping')
            return pred_csv_file

        # assume we are running inference on the standard video
        if video_file is None:
            video_file = os.path.join(
                self.paths.alyx_session_path, 'raw_video_data', self.processed_video_name)

        # assumes commands are run from root directory of tracking-diagnostics
        inference_script = os.path.join(os.getcwd(), 'scripts', 'predict_video.py')

        # construct CLI command
        call_str = \
            f'source {conda_script}; ' + \
            f'conda activate {conda_env}; ' + \
            f'python {inference_script} ' + \
            f'--data_dir {data_dir} ' + \
            f'--model_dir {model_dir} ' + \
            f'--video_file {video_file} ' + \
            f'--pred_csv_file {pred_csv_file} ' + \
            f'--gpu_id {gpu_id} '

        subprocess.run(['/bin/bash', '-c', call_str], check=True)

        return pred_csv_file

    def decode_wrapper(self, results_dir, params, trackers, tracker_name, rng_seed):

        # perform decoding on original eid (-1 entry) and NO pseudo-sessions
        pseudo_ids = np.array([-1])

        # update paths
        params['behfit_path'] = Path(results_dir).joinpath('behavioral')
        params['behfit_path'].mkdir(parents=True, exist_ok=True)
        params['neuralfit_path'] = Path(results_dir)
        params['neuralfit_path'].mkdir(parents=True, exist_ok=True)

        bwm_df = bwm_query(self.one, freeze='2022_10_bwm_release')

        # When merging probes we are interested in eids, not pids
        tmp_df = bwm_df.set_index(['eid', 'subject']).xs(self.eid, level='eid')
        subject = tmp_df.index[0]
        pids = tmp_df['pid'].to_list()  # Select all probes of this session
        probe_names = tmp_df['probe_name'].to_list()

        # create mask
        trials_df, trials_mask = load_trials_and_mask(
            one=self.one, eid=self.eid, sess_loader=self.sess_loader,
            min_rt=params['min_rt'], max_rt=params['max_rt'],
            min_trial_len=params['min_len'], max_trial_len=params['max_len'],
            exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
        params['trials_mask_diagnostics'] = [trials_mask]

        # load spike sorting data, merge across probes
        if data[self.eid]['spikes'] is None:
            clusters_list = []
            spikes_list = []
            for pid, probe_name in zip(pids, probe_names):
                tmp_spikes, tmp_clusters = load_good_units(
                    self.one, pid, eid=self.eid, pname=probe_name)
                tmp_clusters['pid'] = pid
                spikes_list.append(tmp_spikes)
                clusters_list.append(tmp_clusters)
            spikes, clusters = merge_probes(spikes_list, clusters_list)
        else:
            spikes = data[self.eid]['spikes']
            clusters = data[self.eid]['clusters']

        # put everything into the input format fit_eid still expects at this point
        neural_dict = {
            'spk_times': spikes['times'],
            'spk_clu': spikes['clusters'],
            'clu_regions': clusters['acronym'],
            'clu_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
            'clu_df': clusters
        }
        metadata = {
            'subject': subject,
            'eid': self.eid,
            'probe_name': probe_name
        }

        for tracker in trackers:
            print(f'decoding {params["target"]} from {tracker} tracker')
            params['add_to_saving_path'] = f'{tracker}'

            # load target data
            dlc_dict = self.get_target_data(tracker, tracker_name, rng_seed)

            # perform full nested xv decoding
            fit_eid(
                neural_dict=neural_dict,
                trials_df=trials_df,
                trials_mask=trials_mask,
                metadata=metadata,
                pseudo_ids=pseudo_ids,
                dlc_dict=dlc_dict,
                **params)

    @staticmethod
    def _compute_peth(trials, align_event, view, times, feature_vals, feature_name):

        # windows aligned to align_event
        start_window, end_window = plt_window(trials[align_event])
        start_idx = insert_idx(times, start_window)
        end_idx = np.array(start_idx + int(WINDOW_LEN * SAMPLING[view]), dtype='int64')

        # add feature to trials_df
        trials[feature_name] = [
            feature_vals[start_idx[i]:end_idx[i]] for i in range(len(start_idx))]

        # need to expand the series of lists into a dataframe first, for the nan skipping to work
        feedbackType = trials['feedbackType']
        correct = trials[feedbackType == 1][feature_name]
        incorrect = trials[feedbackType == -1][feature_name]
        correct_vals = pd.DataFrame.from_dict(dict(zip(correct.index, correct.values)))
        correct_peth = correct_vals.mean(axis=1)
        incorrect_vals = pd.DataFrame.from_dict(dict(zip(incorrect.index, incorrect.values)))
        incorrect_peth = incorrect_vals.mean(axis=1)

        return {
            'times': np.arange(len(correct_peth)) / SAMPLING[view] + WINDOW_LAG,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'trials': trials,
            'correct_peth': correct_peth.to_numpy(),
            'incorrect_peth': incorrect_peth.to_numpy(),
            'correct_traces': correct_vals.to_numpy(),  # shape (trial_len, n_correct_trials)
            'incorrect_traces': incorrect_vals.to_numpy(),
        }

    def smooth_kalman(self, **kwargs):
        raise NotImplementedError

    def decode(self, **kwargs):
        raise NotImplementedError

    def get_target_data(self, **kwargs):
        raise NotImplementedError


class PupilPipeline(Pipeline):

    def __init__(self, eid, one, likelihood_thr=0.9, base_dir=None):
        super().__init__(
            eid=eid, one=one, view='left', base_dir=base_dir, likelihood_thr=likelihood_thr)

        self.processed_video_name = '_iblrig_leftCamera.cropped_brightened.mp4'
        self.keypoint_names = ['pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r']

        # load pupil data
        self.sess_loader.load_pupil(snr_thresh=0.0)
        self._find_crop_params()

        # load video cap
        self.video = Video()
        self.video.load_video_cap(os.path.join(
            self.paths.alyx_session_path, 'raw_video_data', self.processed_video_name))

        # set paths
        if base_dir:
            self.paths.pupil_csv_dir = os.path.join(base_dir, 'pupil_preds')

    def _find_crop_params(self):
        """Find pupil location in original video for cropping using already-computed markers."""

        # compute location of pupil
        loc = get_pupil_location(self.sess_loader.pose[f'{self.view}Camera'])
        med_x, med_y = np.nanmedian(loc[:, 0]), np.nanmedian(loc[:, 1])

        # assume we'll be dealing with already-upsampled and flipped frames from right camera
        if self.view == 'right':
            # med_x = 2 * (IMG_WIDTH - med_x)
            # med_y *= 2
            raise NotImplementedError

        self.crop_params = {
            'width': 100, 'height': 100, 'left': med_x - 50, 'top': med_y - 50,
            'pupil_x': med_x, 'pupil_y': med_y,
        }

    def pupil_csv_file(self, rng_seed):
        return os.path.join(
            self.paths.pupil_csv_dir, f'pupil.{self.eid}.{self.view}.rng={rng_seed}.csv')

    def smooth_ibl(self, pred_csv_file, pupil_csv_file, tracker_name, overwrite=False):

        # check to make sure predictions exist
        if not os.path.exists(pred_csv_file):
            raise FileNotFoundError(f'could not find prediction csv file {pred_csv_file}')

        # check to see if smoothing has already been run
        if os.path.exists(pupil_csv_file) and not overwrite:
            print(f'{pupil_csv_file} already exists; skipping')
            return pupil_csv_file

        # run smoothing
        dlc_df = get_formatted_df(pred_csv_file, self.keypoint_names, tracker=tracker_name)
        diam_raw = get_pupil_diameter(dlc_df)
        diam_smooth = get_smooth_pupil_diameter(diam_raw, camera=self.view)
        pupil_features = pd.DataFrame(
            {'pupilDiameter_raw': diam_raw, 'pupilDiameter_smooth': diam_smooth})
        pupil_features.to_csv(pupil_csv_file)
        return pupil_csv_file

    def smooth_kalman(
            self, preds_csv_file, latents_csv_file, model_dirs, tracker_name, overwrite=False):

        # check to make sure predictions exist
        all_exist = True
        for rng_seed, model_dir in model_dirs.items():
            csv_file = self.pred_csv_file(rng_seed)
            all_exist &= os.path.exists(csv_file)
        if not all_exist:
            raise FileNotFoundError(f'did not find all prediction files in {model_dirs}')

        # check to see if smoothing has already beeen run
        if os.path.exists(preds_csv_file) and not overwrite:
            print(f'{preds_csv_file} already exists; skipping')
            return preds_csv_file, latents_csv_file

        # run smoothing
        markers_list = []
        diams = []
        for rng_seed, model_dir in model_dirs.items():
            csv_file = self.pred_csv_file(rng_seed)
            markers_tmp = get_formatted_df(csv_file, self.keypoint_names, tracker=tracker_name)
            diams_tmp = get_pupil_diameter(markers_tmp)
            markers_list.append(markers_tmp)
            diams.append(diams_tmp)

        # compute ensemble median
        keys = ['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
                'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']
        ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, keypoints_stack_dict = ensemble_median(
            markers_list, keys)

        # # Kalman Filtering + Smoothing
        # $z_t = (d_t, x_t, y_t)$
        # $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
        # $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

        # ## Set parameters
        # compute center of mass
        pupil_locations = get_pupil_location(keypoints_mean_dict)
        pupil_diameters = get_pupil_diameter(keypoints_mean_dict)
        diameters = []
        for i in range(n_models):
            keypoints_dict = keypoints_stack_dict[i]
            diameter = get_pupil_diameter(keypoints_dict)
            diameters.append(diameter)

        mean_x_obs = np.mean(pupil_locations[:, 0])
        mean_y_obs = np.mean(pupil_locations[:, 1])
        # make the mean zero
        x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

        # latent variables (observed)
        # latent variables - diameter, com_x, com_y
        # z_t_obs = np.vstack((pupil_diameters, x_t_obs, y_t_obs))

        # --------------------------------------
        # Set values for kalman filter
        # --------------------------------------
        # initial state: mean
        m0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0])

        # diagonal: var
        S0 = np.asarray([
            [np.var(pupil_diameters), 0.0, 0.0],
            [0.0, np.var(x_t_obs), 0.0],
            [0.0, 0.0, np.var(y_t_obs)]
        ])

        # state-transition matrix, parameters hand-picked for smoothing purposes
        A = np.asarray([[.9999, 0, 0], [0, .999, 0], [0, 0, .999]])

        # state covariance matrix
        Q = np.asarray([
            [np.var(pupil_diameters) * (1 - (A[0, 0] ** 2)), 0, 0],
            [0, np.var(x_t_obs) * (1 - A[1, 1] ** 2), 0],
            [0, 0, np.var(y_t_obs) * (1 - (A[2, 2] ** 2))]
        ])

        # Measurement function
        C = np.asarray(
            [[0, 1, 0], [-.5, 0, 1], [0, 1, 0], [.5, 0, 1], [.5, 1, 0], [0, 0, 1], [-.5, 1, 0], [0, 0, 1]])

        # placeholder diagonal matrix for ensemble variance
        R = np.eye(8)

        scaled_ensemble_preds = ensemble_preds.copy()
        scaled_ensemble_stacks = ensemble_stacks.copy()
        # subtract COM means from the ensemble predictions
        for i in range(ensemble_preds.shape[1]):
            if i % 2 == 0:
                scaled_ensemble_preds[:, i] -= mean_x_obs
            else:
                scaled_ensemble_preds[:, i] -= mean_y_obs
        # subtract COM means from all the predictions
        for i in range(ensemble_preds.shape[1]):
            if i % 2 == 0:
                scaled_ensemble_stacks[:, :, i] -= mean_x_obs
            else:
                scaled_ensemble_stacks[:, :, i] -= mean_y_obs
        y = scaled_ensemble_preds

        # --------------------------------------
        # perform filtering
        # --------------------------------------
        # do filtering pass with time-varying ensemble variances
        print("filtering...")
        mf, Vf, S = filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
        print("done filtering")

        # --------------------------------------
        # perform smoothing
        # --------------------------------------
        # Do the smoothing step
        print("smoothing...")
        ms, Vs, _ = smooth_backward(y, mf, Vf, S, A, Q, C)
        print("done smoothing")
        # Smoothed posterior over y
        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # --------------------------------------
        # cleanup
        # --------------------------------------
        # save out marker info
        pdindex = make_dlc_pandas_index(keypoint_names)
        processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
        key_pair_list = [['pupil_top_r_x', 'pupil_top_r_y'],
                         ['pupil_right_r_x', 'pupil_right_r_y'],
                         ['pupil_bottom_r_x', 'pupil_bottom_r_y'],
                         ['pupil_left_r_x', 'pupil_left_r_y']]
        pred_arr = []
        for key_pair in key_pair_list:
            pred_arr.append(processed_arr_dict[key_pair[0]])
            pred_arr.append(processed_arr_dict[key_pair[1]])
            var = np.empty(processed_arr_dict[key_pair[0]].shape)
            var[:] = np.nan
            pred_arr.append(var)
        pred_arr = np.asarray(pred_arr)
        df = pd.DataFrame(pred_arr.T, columns=pdindex)
        df.to_csv(preds_csv_file)

        # save out latents info: pupil diam, center of mass
        pred_arr2 = []
        pred_arr2.append(ms[:, 0])
        pred_arr2.append(ms[:, 1] + mean_x_obs)  # add back x mean of pupil location
        pred_arr2.append(ms[:, 2] + mean_y_obs)  # add back y mean of pupil location
        pred_arr2 = np.asarray(pred_arr2)
        arrays = [[tracker_name, tracker_name, tracker_name], ['diameter', 'com_x', 'com_y']]
        pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
        df2 = pd.DataFrame(pred_arr2.T, columns=pd_index2)
        df2.to_csv(latents_csv_file)

        return preds_csv_file, latents_csv_file

    def decode(
            self, date, trackers, tracker_name, rng_seed, align_event='feedback_times',
            results_dir=None):

        if results_dir is None:
            results_dir = self.paths.decoding_dir

        # update params
        params['date'] = date
        params['target'] = 'pupil'
        params['tanh_transform'] = False  # only True for target=='signcont'
        params['align_time'] = align_event
        params['time_window'] = (-0.2, 1.0)  # relative to 'align_time'
        params['binsize'] = 0.02
        params['n_bins_lag'] = 10
        params['n_runs'] = 1
        params['single_region'] = False  # False to combine clusers across all regions
        params['save_binned'] = False
        params['estimator'] = lm.Ridge
        params['hyperparam_grid'] = {'alpha': np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])}
        params['imposter_df'] = None  # need to update this later if we do statistical controls

        self.decode_wrapper(results_dir, params, trackers, tracker_name, rng_seed)

    def get_target_data(self, tracker, tracker_name, rng_seed):
        dlc_dict = {'times': self.sess_loader.pupil['times'].to_numpy(), 'skip': False}
        if tracker == 'dlc':
            dlc_dict['values'] = self.sess_loader.pupil['pupilDiameter_smooth'].to_numpy()
        elif tracker == 'lp':
            filename = self.pupil_csv_file(rng_seed)
            df_tmp = pd.read_csv(filename, header=[0], index_col=0)
            dlc_dict['values'] = df_tmp['pupilDiameter_smooth'].to_numpy()
        elif tracker == 'lp+ks':
            filename = self.kalman_markers_file
            features = pd.read_csv(filename, header=[0, 1], index_col=0)
            dlc_dict['values'] = features.loc[:, (tracker_name, 'diameter')].to_numpy()
        assert dlc_dict['values'].shape[0] == dlc_dict['times'].shape[0]
        return dlc_dict

    def load_markers(self, tracker, tracker_name, rng_seed):
        if tracker == 'dlc':
            dlc_df = self.sess_loader.pose[f'{self.view}Camera']
            # subtract offset
            for kp in self.keypoint_names:
                dlc_df[f'{kp}_x'] = dlc_df[f'{kp}_x'] - self.crop_params['left']
                dlc_df[f'{kp}_y'] = dlc_df[f'{kp}_y'] - self.crop_params['top']
        elif tracker == 'lp':
            dlc_df = get_formatted_df(
                self.pred_csv_file(rng_seed), self.keypoint_names, tracker=tracker_name)
        elif tracker == 'lp+ks':
            dlc_df = get_formatted_df(
                self.kalman_markers_file, self.keypoint_names, tracker='ensemble-kalman_tracker')
        return dlc_df

    def load_pupil_diam(self, tracker, tracker_name, rng_seed, return_smoothed=True):
        if tracker == 'dlc':
            if return_smoothed:
                pupil_diam = self.sess_loader.pupil['pupilDiameter_smooth']
            else:
                pupil_diam = self.sess_loader.pupil['pupilDiameter_raw']
        elif tracker == 'lp':
            features_ = pd.read_csv(self.pupil_csv_file(rng_seed), header=[0], index_col=0)
            if return_smoothed:
                pupil_diam = features_['pupilDiameter_smooth']
            else:
                pupil_diam = features_['pupilDiameter_raw']
        elif tracker == 'lp+ks':
            features_ = pd.read_csv(self.kalman_latents_file, header=[0, 1], index_col=0)
            pupil_diam = features_.loc[:, (tracker_name, 'diameter')]
        return pupil_diam.to_numpy()

    def compute_peth(self, pupil_diam, align_event='feedback_times'):

        # get camera times
        camera_times = self.sess_loader.pupil['times'].to_numpy()

        # get trials data
        cols_to_keep = ['stimOn_times', 'feedback_times', 'feedbackType']
        trials = self.sess_loader.trials.loc[:, cols_to_keep]
        trials = trials.dropna()
        trials = trials.drop(trials[(trials['feedback_times'] - trials['stimOn_times']) > 10].index)

        peth_dict = self._compute_peth(
            trials=trials, align_event=align_event, view=self.view, times=camera_times,
            feature_vals=pupil_diam, feature_name='pupil')

        return peth_dict


class PawPipeline(Pipeline):

    def __init__(self, eid, one, view, likelihood_thr=0.9, base_dir=None):
        super().__init__(
            eid=eid, one=one, view=view, base_dir=base_dir, likelihood_thr=likelihood_thr)

        if view == 'left':
            self.processed_video_name = '_iblrig_leftCamera.downsampled.mp4'
        elif view == 'right':
            self.processed_video_name = '_iblrig_rightCamera.flipped_downsampled.mp4'

        self.keypoint_names = ['paw_l', 'paw_r']

        # load video cap
        self.video = Video()
        self.video.load_video_cap(os.path.join(
            self.paths.alyx_session_path, 'raw_video_data', self.processed_video_name))

        # set paths
        if base_dir:
            self.paths.paw_csv_dir = os.path.join(base_dir, 'paw_preds')

    # TODO
    def smooth_kalman(self, **kwargs):
        raise NotImplementedError

    # TODO
    def decode(self, **kwargs):
        raise NotImplementedError

    # TODO
    def get_target_data(self, **kwargs):
        raise NotImplementedError

    def load_markers(self, tracker, tracker_name, rng_seed):
        if tracker == 'dlc':
            dlc_df = self.sess_loader.pose[f'{self.view}Camera']
            # downsample
            for kp in self.keypoint_names:
                dlc_df[f'{kp}_x'] /= (DS_FACTOR * RESOLUTION[self.view])
                dlc_df[f'{kp}_y'] /= (DS_FACTOR * RESOLUTION[self.view])
        elif tracker == 'lp':
            dlc_df = get_formatted_df(
                self.pred_csv_file(rng_seed), self.keypoint_names, tracker=tracker_name)
        elif tracker == 'lp+ks':
            dlc_df = get_formatted_df(
                self.kalman_markers_file, self.keypoint_names, tracker='ensemble-kalman_tracker')
        return dlc_df

    # TODO
    def load_paw_speed(self):
        pass

    def compute_peth(self, paw_speed, camera_times, align_event='firstMovement_times'):

        # assume everything is aligned to left view timestamps for now
        view = 'left'

        # get trials data
        cols_to_keep = ['stimOn_times', 'feedback_times', 'feedbackType', 'firstMovement_times']
        trials = self.sess_loader.trials.loc[:, cols_to_keep]
        trials = trials.dropna()
        trials = trials.drop(trials[(trials['feedback_times'] - trials['stimOn_times']) > 10].index)

        peth_dict = self._compute_peth(
            trials=trials, align_event=align_event, view=view, times=camera_times,
            feature_vals=paw_speed, feature_name='paw_speed')

        return peth_dict


class Video(object):
    """Simple class for loading videos and timestamps."""

    def __init__(self):

        # opencv video capture
        # type : cv2.VideoCapture object
        self.video_cap = None

        # location of opencv video capture
        # type : str
        self.video_cap_path = None

        # total frames
        # type : int
        self.total_frames = np.nan

        # frame width (pixels)
        # type : int
        self.frame_width = None

        # frame height (pixels)
        # type : int
        self.frame_height = None

    def load_video_cap(self, filepath):
        """Initialize opencv video capture objects from video file.

        Parameters
        ----------
        filepath : str
            absolute location of video (.mp4, .avi)

        """

        # save filepath
        self.video_cap_path = filepath

        # load video cap
        self.video_cap = cv2.VideoCapture(filepath)
        if not self.video_cap.isOpened():
            raise IOError('error opening video file at %s' % filepath)

        # save frame info
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frames_from_idxs(self, idxs):
        """Helper function to load video segments.

        Parameters
        ----------
        idxs : array-like
            frame indices into video

        Returns
        -------
        np.ndarray
            returned frames of shape shape (n_frames, n_channels, y_pix, x_pix)

        """
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, i in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                self.video_cap.set(1, i)
            ret, frame = self.video_cap.read()
            if ret:
                if fr == 0:
                    height, width, _ = frame.shape
                    frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
                frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print(
                    'warning! reached end of video; returning blank frames for remainder of ' +
                    'requested indices')
                break
        return frames


class Paths(object):
    """Class to store paths and allow for easy access."""

    def __init__(self):
        self.alyx_session_path = None


def get_formatted_df(filename, keypoint_names, tracker='heatmap_tracker'):
    dlc_df_ = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
    dlc_df = {}
    for kp in keypoint_names:
        for f in ['x', 'y', 'likelihood']:
            dlc_df[f'{kp}_{f}'] = dlc_df_.loc[:, (tracker, kp, f)]
    return pd.DataFrame(dlc_df, index=dlc_df_.index)


def get_pupil_location(dlc_df):
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
    t = np.vstack((dlc_df['pupil_top_r_x'], dlc_df['pupil_top_r_y'])).T / s
    b = np.vstack((dlc_df['pupil_bottom_r_x'], dlc_df['pupil_bottom_r_y'])).T / s
    l = np.vstack((dlc_df['pupil_left_r_x'], dlc_df['pupil_left_r_y'])).T / s
    r = np.vstack((dlc_df['pupil_right_r_x'], dlc_df['pupil_right_r_y'])).T / s
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
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_x
        else:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_y
    return processed_arr_dict


def make_dlc_pandas_index(keypoint_names):
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % 'ensemble-kalman'], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex
