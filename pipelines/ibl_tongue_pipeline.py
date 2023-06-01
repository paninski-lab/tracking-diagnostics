"""Pipeline function that will perform the following steps of the ibl pupil pipeline:

1. preprocess videos using ffmpeg OR re-encode already-processed mp4 videos using yuv420
2. run inference on videos
3a. smooth predictions using ibl heuristics
3b. smooth predictions using ensemble kalman smoother
4. decode pupil diameter from neural activity

NOTE: this must be run from inside tracking-diagnostics

"""

from one.api import ONE
import os

from diagnostics.paper_ibl import TonguePipeline

# ----------------------------
# pipeline user options
# ----------------------------
# location of labeled dataset
data_dir = '/media/mattw/behavior/pose-estimation-data-final/ibl-tongue'

# location of pipeline outputs
base_dir = '/media/mattw/behavior/results/pose-estimation/ibl-tongue/ensembling-expts'
"""directory structure

base_dir
├── decoding
├── figs
├── kalman_outputs
├── models
└── video_preds
    ├── <eid_0>.left.rng=0.csv
    ├── ...
    └── <eid_n>.left.rng=m.csv

"""

gpu_id = 0
rng_seed = 0  # choose ensemble member for single-model analyses (decoding)
tracker_name = 'heatmap_mhcrnn_tracker'
pipe_kwargs = {
    'download_data': {  # download data from flatiron
        'run': True, 'kwargs': {}},
    'preprocess_video': {  # process and reencode video with ffmpeg
        'run': False, 'kwargs': {'overwrite': False, 'mp4_file': None}},
    'reencode_video': {  # reencode already ffmpeg-processed video
        'run': True, 'kwargs': {'overwrite': False}},
    'infer_video': {
        'run': True, 'kwargs': {'overwrite': False, 'gpu_id': gpu_id}},
    'smooth_kalman': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
}

# ----------------------------
# tongue eids
# ----------------------------
# labeled vids from first round of DLC
# eids = [
#     # 'dd875d90-ac39-4c06-b4ec-ede3340f39af',
#     'a1153fb1-4cbd-4ad1-8464-48d19580c0a0',
#     '76b5e01f-b1fb-4d90-9032-6b4b44643852',
#     '433f172f-cdd6-468e-b4c2-5ae1a360b12c',
#     'b2e58955-0c6c-4892-a749-f30fd2df50df',
#     'cae0b6a5-5979-453f-a8ff-53d5bc4cae0f',
#     '15be225a-55ae-4ab4-acf6-2dd41ed91ba9',
#     '5756d1b2-4211-4f5f-b1e5-ff032804dc22',
#     '2a2f1b6b-a952-4ef5-8e39-86d486d2bb00',
#     '20590ae2-db24-4fed-b4d7-249827247114',
#     'be57d6f9-d820-4059-8a9f-1e817fd407dc',
#     '1d609dde-6e95-46fa-b128-89248e57d868',
#     'aa585952-1208-43e3-bae4-bf6d88b78a9d',
#     '82592038-2d52-416f-8fbc-e17f2042790e',
#     '22492322-4107-4cd3-870d-01b899394524',
#     'b384b3f2-e379-4010-be01-c8c3784faeeb',
#     '78704a74-84c3-4134-b79b-5547655d2440',
#     '3850497c-4c92-4ace-b9d7-995b40e6e447',
#     'fd9ed7c8-ad23-480f-bd65-d2fcb2424570',
#     '0c539f6a-b511-4553-b952-fe1df10677bd',
#     'a5a5e1a0-646f-4f56-b72b-0259f9d7c249',
#     '8181ca89-42b7-4ff2-a0f9-4e609d6f5c67',
# ]

# pipeline testing
eids = [
    '15948667-747b-4702-9d53-354ac70e9119',
    # 'aad23144-0e52-4eac-80c5-c4ee2decb198',
]


# hard code cropping params for these eids
# CROP_PARAMS = {
#     'dd875d90-ac39-4c06-b4ec-ede3340f39af': {'width': 160, 'height': 160, 'left': 294, 'top': 246},
#     'a1153fb1-4cbd-4ad1-8464-48d19580c0a0': {'width': 160, 'height': 160, 'left': 402, 'top': 300},
#     '76b5e01f-b1fb-4d90-9032-6b4b44643852': {'width': 160, 'height': 160, 'left': 214, 'top': 348},
#     '433f172f-cdd6-468e-b4c2-5ae1a360b12c': {'width': 160, 'height': 160, 'left': 250, 'top': 424},
#     'b2e58955-0c6c-4892-a749-f30fd2df50df': {'width': 160, 'height': 160, 'left': 132, 'top': 334},
#     'cae0b6a5-5979-453f-a8ff-53d5bc4cae0f': {'width': 160, 'height': 160, 'left': 172, 'top': 432},
#     '15be225a-55ae-4ab4-acf6-2dd41ed91ba9': {'width': 160, 'height': 160, 'left': 192, 'top': 276},
#     '5756d1b2-4211-4f5f-b1e5-ff032804dc22': {'width': 160, 'height': 160, 'left': 218, 'top': 412},
#     '2a2f1b6b-a952-4ef5-8e39-86d486d2bb00': {'width': 160, 'height': 160, 'left': 192, 'top': 298},
#     '20590ae2-db24-4fed-b4d7-249827247114': {'width': 160, 'height': 160, 'left': 282, 'top': 288},
#     'be57d6f9-d820-4059-8a9f-1e817fd407dc': {'width': 160, 'height': 160, 'left': 222, 'top': 230},
#     '1d609dde-6e95-46fa-b128-89248e57d868': {'width': 160, 'height': 160, 'left': 196, 'top': 394},
#     'aa585952-1208-43e3-bae4-bf6d88b78a9d': {'width': 160, 'height': 160, 'left': 174, 'top': 248},
#     '82592038-2d52-416f-8fbc-e17f2042790e': {'width': 160, 'height': 160, 'left': 274, 'top': 244},
#     '22492322-4107-4cd3-870d-01b899394524': {'width': 160, 'height': 160, 'left': 166, 'top': 312},
#     'b384b3f2-e379-4010-be01-c8c3784faeeb': {'width': 160, 'height': 160, 'left': 184, 'top': 366},
#     '78704a74-84c3-4134-b79b-5547655d2440': {'width': 160, 'height': 160, 'left': 374, 'top': 326},
#     '3850497c-4c92-4ace-b9d7-995b40e6e447': {'width': 160, 'height': 160, 'left': 160, 'top': 322},
#     'fd9ed7c8-ad23-480f-bd65-d2fcb2424570': {'width': 160, 'height': 160, 'left': 286, 'top': 306},
#     '0c539f6a-b511-4553-b952-fe1df10677bd': {'width': 160, 'height': 160, 'left': 264, 'top': 358},
#     'a5a5e1a0-646f-4f56-b72b-0259f9d7c249': {'width': 160, 'height': 160, 'left': 234, 'top': 220},
#     '8181ca89-42b7-4ff2-a0f9-4e609d6f5c67': {'width': 160, 'height': 160, 'left': 292, 'top': 286},
# }

# ----------------------------
# lp model directories
# ----------------------------
model_dirs = {
    '0': '/media/mattw/behavior/results/pose-estimation/ibl-tongue/2023-06-01/14-45-41',
    # '1': None,
    # '2': None,
    # '3': None,
    # '4': None,
}

# ----------------------------
# run pipeline
# ----------------------------
one = ONE()

error_log = {}
for e, eid in enumerate(eids):

    print(f'eid {e}: {eid}')

    # try:

    pipe = TonguePipeline(
        eid=eid, one=one, base_dir=base_dir,
        allow_trial_fail=False, load_dlc=True,
        # for sessions that have no DLC traces
        # allow_trial_fail=True, load_dlc=False,
        # crop_params=CROP_PARAMS[eid],
    )
    print(pipe.paths.alyx_session_path)

    # preprocess video into correct size/shape for lightning pose network
    if pipe_kwargs['download_data']['run']:
        pipe.download_data(**pipe_kwargs['download_data']['kwargs'])

    # preprocess video into correct size/shape for lightning pose network
    if pipe_kwargs['preprocess_video']['run']:
        pipe.preprocess_video(**pipe_kwargs['preprocess_video']['kwargs'])

    # re-encode video into yuv420p format need for lightning pose
    if pipe_kwargs['reencode_video']['run']:
        pipe.reencode_video(**pipe_kwargs['reencode_video']['kwargs'])

    # run inference on video for each ensemble member
    if pipe_kwargs['infer_video']['run']:
        for rng_seed, model_dir in model_dirs.items():
            pred_csv_file = pipe.pred_csv_file(rng_seed)
            pipe.infer_video(
                model_dir=model_dir, data_dir=data_dir, pred_csv_file=pred_csv_file,
                **pipe_kwargs['infer_video']['kwargs'])

    # # smooth results using ensemble kalman smoother
    # if pipe_kwargs['smooth_kalman']['run']:
    #     pipe.smooth_kalman(
    #         preds_csv_file=pipe.kalman_markers_file, latents_csv_file=pipe.kalman_latents_file,
    #         model_dirs=model_dirs, **pipe_kwargs['smooth_kalman']['kwargs'])

    # except Exception as exception:
    #     error_log[eid] = exception

print('tongue pipeline finished')
for key, val in error_log.items():
    print(f'{key}: {val}\n')
