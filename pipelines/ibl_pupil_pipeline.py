"""Pipeline function that will perform the following steps of the ibl pupil pipeline:

1. re-encode mp4 videos using yuv420
2. run inference on videos
3a. smooth predictions using ibl heuristics
3b. smooth predictions using ensemble kalman smoother
4. decode pupil diameter from neural activity

"""

from one.api import ONE
import os

from diagnostics.paper_ibl import PupilPipeline

# ----------------------------
# pipeline user options
# ----------------------------
data_dir = '/media/mattw/behavior/pose-estimation-data-final/ibl-pupil'
base_dir = '/media/mattw/behavior/results/pose-estimation/ibl-pupil-fixed/ensembling-expts'
"""directory structure

base_dir
├── decoding
├── figs
├── kalman_outputs
├── models
├── pupil_preds
│   ├── pupil.<eid_0>.left.rng=0.csv
│   ├── ...
│   └── pupil.<eid_n>.left.rng=m.csv
└── video_preds
    ├── <eid_0>.left.rng=0.csv
    ├── ...
    └── <eid_n>.left.rng=m.csv

"""

gpu_id = 0
rng_seed = 0  # choose ensemble member for single-model analyses
tracker_name = 'heatmap_mhcrnn_tracker'
pipe_kwargs = {
    'reencode_video': {
        'run': False, 'kwargs': {'overwrite': False}},
    'infer_video': {
        'run': False, 'kwargs': {'overwrite': False, 'gpu_id': gpu_id}},
    'smooth_ibl': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'smooth_kalman': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'decode': {
        'run': False,
        'kwargs': {
            # 'overwrite': False,
            'tracker_name': tracker_name,
            'trackers': ['lp', 'lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            'rng_seed': rng_seed,  # for 'lp' tracker
            'date': '02-02-2023',
        }},
}

# ----------------------------
# pupil eids
# ----------------------------
eids = [
    'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
    '5285c561-80da-4563-8694-739da92e5dd0',
    '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',
    '8928f98a-b411-497e-aa4b-aa752434686d',
    '3d59aa1a-b4ba-47fe-b9cf-741b5fdb0c7b',
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
    '5d6aa933-4b00-4e99-ae2d-5003657592e9',
    'a92c4b1d-46bd-457e-a1f4-414265f0e2d4',
    '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00',
    '9468fa93-21ae-4984-955c-e8402e280c83',
    '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
    'b658bc7d-07cd-4203-8a25-7b16b549851b',
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
    '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
    '5bcafa14-71cb-42fa-8265-ce5cda1b89e0',
    # '8207abc6-6b23-4762-92b4-82e05bed5143',  # issue
    '6364ff7f-6471-415a-ab9e-632a12052690',
    'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
    'd0c91c3c-8cbb-4929-8657-31f18bffc294',
    '30e5937e-e86a-47e6-93ae-d2ae3877ff8e',
    '931a70ae-90ee-448e-bedb-9d41f3eda647',
    '78b4fff5-c5ec-44d9-b5f9-d59493063f00',
    'c6db3304-c906-400c-aa0f-45dd3945b2ea',
    '9fe512b8-92a8-4642-83b6-01158ab66c3c',
    'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
    'aa20388b-9ea3-4506-92f1-3c2be84b85db',
    '0b7ee1b6-42db-46cd-a465-08f531366187',
    'd2f5a130-b981-4546-8858-c94ae1da75ff',
    '768a371d-7e88-47f8-bf21-4a6a6570dd6e',
    '948fd27b-507b-41b3-bdf8-f9f5f0af8e0b',
    '752456f3-9f47-4fbf-bd44-9d131c0f41aa',
    '81a1dca0-cc90-47c5-afe3-c277319c47c8',
    'a4747ac8-6a75-444f-b99b-696fff0243fd',
    '6668c4a0-70a4-4012-a7da-709660971d7a',
    '1ca83b26-30fc-4350-a616-c38b7d00d240',
    # 'b81e3e11-9a60-4114-b894-09f85074d9c3',  # issue
    '7416f387-b302-4ca3-8daf-03b585a1b7ec',
    '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
    'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
    '952870e5-f2a7-4518-9e6d-71585460f6fe',
    '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
    '3f6e25ae-c007-4dc3-aa77-450fd5705046',
    '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
    '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
    'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
    '821f1883-27f3-411d-afd3-fb8241bbc39a',
    '75b6b132-d998-4fba-8482-961418ac957d',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',
]

# ----------------------------
# lp model directories
# ----------------------------
model_dirs = {
    '0': os.path.join(base_dir, 'models/functional-nightingale-5302-exp0/outputs/2023-01-26/18-47-00'),
    '1': os.path.join(base_dir, 'models/functional-nightingale-5302-exp1/outputs/2023-01-26/18-46-53'),
    '2': os.path.join(base_dir, 'models/functional-nightingale-5302-exp2/outputs/2023-01-26/18-46-53'),
    '3': os.path.join(base_dir, 'models/functional-nightingale-5302-exp3/outputs/2023-01-26/18-47-04'),
    '4': os.path.join(base_dir, 'models/functional-nightingale-5302-exp4/outputs/2023-01-26/18-46-55'),
}

# ----------------------------
# run pipeline
# ----------------------------
one = ONE()

error_log = {}
for e, eid in enumerate(eids):

    pipe = PupilPipeline(eid=eid, one=one, base_dir=base_dir)
    print(f'eid {e}: {eid}')
    print(pipe.paths.alyx_session_path)

    try:
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

        # smooth results of each ensemble member using ibl heuristics (savitzy-golay filter)
        if pipe_kwargs['smooth_ibl']['run']:
            for rng_seed, model_dir in model_dirs.items():
                pred_csv_file = pipe.pred_csv_file(rng_seed)
                pupil_csv_file = pipe.pupil_csv_file(rng_seed)
                pipe.smooth_ibl(
                    pred_csv_file=pred_csv_file, pupil_csv_file=pupil_csv_file,
                    **pipe_kwargs['smooth_ibl']['kwargs'])

        # smooth results using ensemble kalman smoother
        if pipe_kwargs['smooth_kalman']['run']:
            pipe.smooth_kalman(
                preds_csv_file=pipe.kalman_markers_file, latents_csv_file=pipe.kalman_latents_file,
                model_dirs=model_dirs, **pipe_kwargs['smooth_kalman']['kwargs'])

        # decode pupil diameter from simultaneously recorded neural activity
        if pipe_kwargs['decode']['run']:
            pipe.decode(**pipe_kwargs['decode']['kwargs'])

    except Exception as exception:
        error_log[eid] = exception


print('pipeline finished')
print(error_log)
