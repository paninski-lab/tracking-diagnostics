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
from neurodsp.smooth import smooth_interpolate_savgol
import numpy as np
from one.api import ONE
import os
import pandas as pd
from pathlib import Path
from scipy import interpolate
from sklearn import linear_model as lm
import subprocess

from diagnostics.ensemble_kalman_filter import ensemble_kalman_smoother_pupil
from diagnostics.ensemble_kalman_filter import ensemble_kalman_smoother_paw_asynchronous
from diagnostics.ensemble_kalman_filter import get_pupil_location


# camera constants (from Michael Schartner)
FOCAL_LENGTH_MM = 16
SENSOR_SIZE = 12.7
IMG_WIDTH = 640
IMG_HEIGHT = 512

# downsampling factor used for paw inference
DS_FACTOR = 5

# peth computation/plotting info
WINDOW_LAG = -0.4

# local info
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
        self.sess_loader.load_pose(views=['left', 'right'], likelihood_thr=likelihood_thr)

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

        # perform decoding on original eid (-1 entry) and NO pseudo-sessions (positive ints)
        pseudo_ids = np.array([-1])

        # update paths
        params['behfit_path'] = Path(results_dir).joinpath('behavioral')
        params['behfit_path'].mkdir(parents=True, exist_ok=True)
        params['neuralfit_path'] = Path(results_dir)
        params['neuralfit_path'].mkdir(parents=True, exist_ok=True)

        # load all bwm session info
        bwm_df = bwm_query(self.one, freeze='2022_10_bwm_release')

        # when merging probes we are interested in eids, not pids
        tmp_df = bwm_df.set_index(['eid', 'subject']).xs(self.eid, level='eid')
        subject = tmp_df.index[0]
        pids = tmp_df['pid'].to_list()  # Select all probes of this session
        probe_names = tmp_df['probe_name'].to_list()

        # create trial mask
        trials_df, trials_mask = load_trials_and_mask(
            one=self.one, eid=self.eid, sess_loader=self.sess_loader,
            min_rt=params['min_rt'], max_rt=params['max_rt'],
            min_trial_len=params['min_len'], max_trial_len=params['max_len'],
            exclude_nochoice=True, exclude_unbiased=params['exclude_unbiased_trials'])
        params['trials_mask_diagnostics'] = [trials_mask]

        # load spike sorting data, merge across probes
        clusters_list = []
        spikes_list = []
        for pid, probe_name in zip(pids, probe_names):
            tmp_spikes, tmp_clusters = load_good_units(
                self.one, pid, eid=self.eid, pname=probe_name)
            tmp_clusters['pid'] = pid
            spikes_list.append(tmp_spikes)
            clusters_list.append(tmp_clusters)
        spikes, clusters = merge_probes(spikes_list, clusters_list)

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
            dlc_dict = self.get_target_data(tracker, tracker_name, rng_seed, params['target'])

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
            'start_idxs': start_idx,
            'end_idxs': end_idx,
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

        # load markers and compute initial pupil diameter
        markers_list = []
        for rng_seed, model_dir in model_dirs.items():
            csv_file = self.pred_csv_file(rng_seed)
            markers_tmp = get_formatted_df(csv_file, self.keypoint_names, tracker=tracker_name)
            markers_list.append(markers_tmp)

        # run ks
        df_dict = ensemble_kalman_smoother_pupil(markers_list, tracker_name)

        # save smoothed markers
        os.makedirs(os.path.dirname(preds_csv_file), exist_ok=True)
        df_dict['markers_df'].to_csv(preds_csv_file)

        # save latents info: pupil diam, center of mass
        os.makedirs(os.path.dirname(latents_csv_file), exist_ok=True)
        df_dict['latents_df'].to_csv(latents_csv_file)

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

    def decode(
            self, paw, date, trackers, tracker_name, rng_seed, align_event='firstMovement_times',
            results_dir=None):

        if results_dir is None:
            results_dir = self.paths.decoding_dir

        # update params
        params['date'] = date
        params['target'] = f'{self.view}_cam_{paw}'
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

    def get_target_data(self, tracker, tracker_name, rng_seed, target):
        """Load paw speed."""

        # select view
        if tracker == 'dlc':
            camera = self.view
        elif tracker == 'lp':
            camera = self.view
        elif tracker == 'lp+ks':
            # kalman outputs always at resolution of left camera
            camera = 'left'

        # load times
        times = self.sess_loader.pose[f'{camera}Camera']['times'].to_numpy()
        # load poses
        poses = self.load_markers(tracker, tracker_name, rng_seed)

        len_times = times.shape[0]
        len_poses = poses.shape[0]
        if len_times != len_poses:
            print(f'WARNING! timestamp length {len_times}, pose length, {len_poses}')
            if len_poses > len_times:
                raise ValueError('Poses longer than timestamps, do not know how to proceed')
            times = times[:len_poses]

        # NOTE: this feature references the name in the pose dataframes; the actual limb this
        # refers to is limb-dependent
        # view=left, feature=paw_l: left paw
        # view=left, feature=paw_r: right paw
        # view=right, feature=paw_l: right paw
        # view=right, feature=paw_r: left paw
        if target.find('paw_l') > -1:
            feature = 'paw_l'
        elif target.find('paw_r') > -1:
            feature = 'paw_r'
        else:
            raise NotImplementedError

        # # linearly interpolate through dropped points
        # for coord in ['x', 'y']:
        #     c = poses[f'{feature}_{coord}'].to_numpy()
        #     if np.any(np.isnan(c)):
        #         mask = ~np.isnan(c)
        #         ifcn = interpolate.interp1d(times[mask], c[mask], fill_value="extrapolate")
        #         tmp = ifcn(times)
        #         poses.loc[:, f'{feature}_{coord}'] = tmp

        # apply light smoothing
        window = 13 if camera == 'right' else 7
        x_raw = poses[f'{feature}_x'].to_numpy()
        x_smooth = smooth_interpolate_savgol(x_raw, window=window, order=3, interp_kind='linear')
        y_raw = poses[f'{feature}_y'].to_numpy()
        y_smooth = smooth_interpolate_savgol(y_raw, window=window, order=3, interp_kind='linear')
        poses = pd.DataFrame({f'{feature}_x': x_smooth, f'{feature}_y': y_smooth})

        # compute speed for desired feature
        vals = get_speed(poses, times, camera=camera, feature=feature)

        return {'times': times, 'values': vals, 'skip': False}

    def load_markers(self, tracker, tracker_name, rng_seed):
        if tracker == 'dlc':
            dlc_df = self.sess_loader.pose[f'{self.view}Camera']
            # downsample
            for kp in self.keypoint_names:
                dlc_df[f'{kp}_x'] /= (DS_FACTOR * RESOLUTION[self.view])
                dlc_df[f'{kp}_y'] /= (DS_FACTOR * RESOLUTION[self.view])
            # flip horizontally
            if self.view == 'right':
                for kp in self.keypoint_names:
                    dlc_df[f'{kp}_x'] = IMG_WIDTH / DS_FACTOR - dlc_df[f'{kp}_x']
        elif tracker == 'lp':
            dlc_df = get_formatted_df(
                self.pred_csv_file(rng_seed), self.keypoint_names, tracker=tracker_name)
        elif tracker == 'lp+ks':
            dlc_df = get_formatted_df(
                self.kalman_markers_file, self.keypoint_names, tracker='ensemble-kalman_tracker')
        return dlc_df

    def compute_peth(self, paw_speed, camera_times, align_event='firstMovement_times'):

        # assume everything is aligned to left view timestamps for now
        view = 'left'

        # get trials data
        cols_to_keep = [
            'stimOn_times', 'feedback_times', 'feedbackType', 'firstMovement_times', 'choice']
        trials = self.sess_loader.trials.loc[:, cols_to_keep]
        trials = trials.dropna()
        trials = trials.drop(trials[(trials['feedback_times'] - trials['stimOn_times']) > 10].index)

        peth_dict = self._compute_peth(
            trials=trials, align_event=align_event, view=view, times=camera_times,
            feature_vals=paw_speed, feature_name='paw_speed')

        return peth_dict


class MultiviewPawPipeline(object):

    def __init__(self, eid, one, likelihood_thr=0.9, base_dir=None):

        self.eid = eid
        self.one = one

        self.pipes = {
            v: PawPipeline(
                eid=eid, one=one, view=v, likelihood_thr=likelihood_thr, base_dir=base_dir
            ) for v in ['left', 'right']}

        self.keypoint_names = ['paw_l', 'paw_r']
        self.views = ['left', 'right']

        # copy paths from one of the single-view pipeline objects
        self.paths = self.pipes['left'].paths

    def kalman_latents_file(self, view):
        return os.path.join(
            self.paths.kalman_save_dir, f'latents.kalman_smoothed.{self.eid}.{view}.csv')

    def kalman_markers_file(self, view):
        return os.path.join(
            self.paths.kalman_save_dir, f'markers.kalman_smoothed.{self.eid}.{view}.csv')

    def timestamps_file(self, view):
        return os.path.join(self.paths.alyx_session_path, 'alf', f'_ibl_{view}Camera.times.npy')

    def smooth_kalman(
            self, preds_csv_files, model_dirs, tracker_name, timestamp_files=None,
            overwrite=False):

        # check to make sure predictions exist
        all_exist = True
        for rng_seed, model_dir in model_dirs.items():
            for view in self.views:
                csv_file = self.pipes[view].pred_csv_file(rng_seed)
                all_exist &= os.path.exists(csv_file)
        if not all_exist:
            raise FileNotFoundError(f'did not find all prediction files in {model_dirs}')

        # check to see if smoothing has already beeen run
        all_exist = True
        for view, file in preds_csv_files.items():
            if os.path.exists(file) and not overwrite:
                pass
            else:
                all_exist = False
        if all_exist:
            print(f'{file} already exists; skipping')
            return preds_csv_files

        # collect markers across ensemble members from both views
        markers_list_l_cam = []
        markers_list_r_cam = []
        for rng_seed, model_dir in model_dirs.items():
            # load markers from this ensemble member
            csv_file_l = self.pipes['left'].pred_csv_file(rng_seed)
            markers_tmp_l_cam = get_formatted_df(
                csv_file_l, self.keypoint_names, tracker=tracker_name)
            csv_file_r = self.pipes['right'].pred_csv_file(rng_seed)
            markers_tmp_r_cam = get_formatted_df(
                csv_file_r, self.keypoint_names, tracker=tracker_name)

            # append to ensemble list
            markers_list_l_cam.append(markers_tmp_l_cam)
            # switch right camera paws
            columns = {
                'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
                'paw_l_likelihood': 'paw_r_likelihood',
                'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
                'paw_r_likelihood': 'paw_l_likelihood'
            }
            markers_tmp_r_cam = markers_tmp_r_cam.rename(columns=columns)
            # reorder
            markers_tmp_r_cam = markers_tmp_r_cam.loc[:, columns.keys()]
            markers_list_r_cam.append(markers_tmp_r_cam)

        # collect timestamps from both views
        if timestamp_files is None:
            # load from ibl database
            timestamps_l_cam = np.load(self.timestamps_file('left'))
            timestamps_r_cam = np.load(self.timestamps_file('right'))
        else:
            # assume dict
            timestamps_l_cam = np.load(timestamp_files['left'])
            timestamps_r_cam = np.load(timestamp_files['right'])

        if timestamps_l_cam.shape[0] != markers_list_l_cam[0].shape[0]:
            raise ValueError('left camera timestamp misalignment')
        if timestamps_r_cam.shape[0] != markers_list_r_cam[0].shape[0]:
            raise ValueError('right camera timestamp misalignment')

        # run ks
        df_dict = ensemble_kalman_smoother_paw_asynchronous(
            markers_list_left_cam=markers_list_l_cam,
            markers_list_right_cam=markers_list_r_cam,
            timestamps_left_cam=timestamps_l_cam,
            timestamps_right_cam=timestamps_r_cam,
            keypoint_names=self.keypoint_names,
        )

        # save smoothed markers from left view
        os.makedirs(os.path.dirname(preds_csv_files['left']), exist_ok=True)
        df_dict['left_df'].to_csv(preds_csv_files['left'])

        # save smoothed markers from right view
        os.makedirs(os.path.dirname(preds_csv_files['right']), exist_ok=True)
        df_dict['right_df'].to_csv(preds_csv_files['right'])

        return preds_csv_files


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


# -----------------------
# general funcs
# -----------------------
def get_formatted_df(filename, keypoint_names, tracker='heatmap_tracker'):
    dlc_df_ = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
    dlc_df = {}
    for kp in keypoint_names:
        for f in ['x', 'y', 'likelihood']:
            dlc_df[f'{kp}_{f}'] = dlc_df_.loc[:, (tracker, kp, f)]
    return pd.DataFrame(dlc_df, index=dlc_df_.index)


def get_speed(dlc, dlc_t, camera, feature='paw_r'):
    """
    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """
    x = dlc[f'{feature}_x']
    y = dlc[f'{feature}_y']

    # get speed in px/sec [half res]
    s = ((np.diff(x) ** 2 + np.diff(y) ** 2) ** .5) * SAMPLING[camera]

    dt = np.diff(dlc_t)
    tv = dlc_t[:-1] + dt / 2

    # interpolate over original time scale
    if tv.size > 1:
        ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
        return ifcn(dlc_t)
