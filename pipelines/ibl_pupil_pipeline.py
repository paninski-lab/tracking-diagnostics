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

gpu_id = 1
rng_seed = 0  # choose ensemble member for single-model analyses (decoding)
tracker_name = 'heatmap_mhcrnn_tracker'
pipe_kwargs = {
    'download_data': {  # download data from flatiron
        'run': False, 'kwargs': {}},
    'preprocess_video': {  # process and reencode video with ffmpeg
        'run': False, 'kwargs': {'overwrite': False, 'mp4_file': None}},
    'reencode_video': {  # reencode already ffmpeg-processed video
        'run': False, 'kwargs': {'overwrite': False}},
    'infer_video': {
        'run': False, 'kwargs': {'overwrite': False, 'gpu_id': gpu_id}},
    'smooth_ibl': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'smooth_kalman': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'decode': {
        'run': True,
        'kwargs': {
            # 'overwrite': False,  # not supported yet
            'tracker_name': tracker_name,
            'trackers': ['dlc', 'lp', 'lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            'rng_seed': rng_seed,  # for 'lp' tracker
            # -- 20 ms bins
            # 'date': '02-02-2023',
            # 'binsize': 0.02,
            # 'n_bins_lag': 10,
            # -- 40 ms bins
            # 'date': '03-28-2023',
            # 'binsize': 0.04,
            # 'n_bins_lag': 5,
            # -- 50 ms bins
            'date': '03-29-2023',
            'binsize': 0.05,
            'n_bins_lag': 5
            # -- 100 ms bins
            # 'date': '03-30-2023',
            # 'binsize': 0.1,
            # 'n_bins_lag': 3,
        }},
}

# ----------------------------
# pupil eids
# ----------------------------
# litpose paper
eids = [
    # -------------------------
    # original sessions
    # -------------------------
    # no issues
    'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
    '5285c561-80da-4563-8694-739da92e5dd0',
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
    '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
    'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
    '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
    '3f6e25ae-c007-4dc3-aa77-450fd5705046',
    '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
    '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
    'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
    '821f1883-27f3-411d-afd3-fb8241bbc39a',
    '75b6b132-d998-4fba-8482-961418ac957d',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',
    # issues
    # '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',  # left/right timestamp issue
    # '8207abc6-6b23-4762-92b4-82e05bed5143',  # dlc issue?
    # 'b81e3e11-9a60-4114-b894-09f85074d9c3',  # timestamp issue
    # '7416f387-b302-4ca3-8daf-03b585a1b7ec',  # left/right timestamp issue
    # '952870e5-f2a7-4518-9e6d-71585460f6fe',  # left/right timestamp issue
    #
    # -------------------------
    # additional repro sessions (22)
    # -------------------------
    # sync good
    '034e726f-b35f-41e0-8d6c-a22cc32391fb',  # FELIX
    '56b57c38-2699-4091-90a8-aba35103155e',  # in paw labeled data
    'ee40aece-cffd-4edb-a4b6-155f158c666a',
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
    'dda5fc59-f09a-4256-9fb5-66c67667a466',
    # '4b00df29-3769-43be-bb40-128b1cba6d35',  # bad video quality
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # in paw labeled data
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
    '51e53aff-1d5d-4182-a684-aba783d50ae5',
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
    '3638d102-e8b6-4230-8742-e548cd87a949',
    '88224abb-5746-431f-9c17-17d7ef806e6a',
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # in litpose paw list already
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
    '3f859b5c-e73a-4044-b49e-34bb81e96715',
    'b22f694e-4a34-4142-ab9d-2556c3487086',
    '746d1902-fa59-4cab-b0aa-013be36060d5',  # in litpose paw list already
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',
    # sync ok
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # off by one; in litpose paw list already
    'ecb5520d-1358-434c-95ec-93687ecd1396',  # off by one
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # off by one?
]

# for felix/prior paper
# eids = [
#     # '07dc4b76-5b93-4a03-82a0-b3d9cc73f412',  # timestamp issue
#     # '6274dda8-3a59-4aa1-95f8-a8a549c46a26',  # timestamp issue
#     '034e726f-b35f-41e0-8d6c-a22cc32391fb',
# ]

# for repro-ephys paper
# eids = [
#     # sync good
#     '56b57c38-2699-4091-90a8-aba35103155e',  # in paw labeled data
#     '41872d7f-75cb-4445-bb1a-132b354c44f0',  # not in BWM dataset
#     'ee40aece-cffd-4edb-a4b6-155f158c666a',
#     '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # not in BWM dataset
#     'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
#     'dda5fc59-f09a-4256-9fb5-66c67667a466',
#     '4b00df29-3769-43be-bb40-128b1cba6d35',  # bad video quality
#     'db4df448-e449-4a6f-a0e7-288711e7a75a',  # in pupil/paw labeled data
#     'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
#     'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # in paw labeled data
#     '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
#     '51e53aff-1d5d-4182-a684-aba783d50ae5',
#     '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
#     '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # in litpose pupil list already
#     'f140a2ec-fd49-4814-994a-fe3476f14e66',  # not in BWM dataset
#     '3638d102-e8b6-4230-8742-e548cd87a949',
#     '88224abb-5746-431f-9c17-17d7ef806e6a',
#     'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # in litpose paw list already
#     'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
#     '3f859b5c-e73a-4044-b49e-34bb81e96715',
#     '754b74d5-7a06-4004-ae0c-72a10b6ed2e6',  # not in BWM dataset; bad cam positions
#     'd9f0c293-df4c-410a-846d-842e47c6b502',  # in paw labeled data; not in BWM dataset
#     'b22f694e-4a34-4142-ab9d-2556c3487086',
#     '746d1902-fa59-4cab-b0aa-013be36060d5',  # in litpose paw list already
#     '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
#     'aad23144-0e52-4eac-80c5-c4ee2decb198',  # in pupil/paw labeled data
#     '0802ced5-33a3-405e-8336-b65ebc5cb07c',
#     # sync ok
#     '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # off by one; in litpose paw list already
#     'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # off by one? not in BWM dataset
#     'ecb5520d-1358-434c-95ec-93687ecd1396',  # off by one
#     '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # off by one?
#     'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # off by one or two; not in BWM dataset
#     '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # off by one?; not in BWM dataset
#     '8928f98a-b411-497e-aa4b-aa752434686d',  # off by one; in litpose pupil list already
#     # sync BAD
#     # '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # left cam ok (in litpose pupil list)
#     # '862ade13-53cd-4221-a3fa-dda8643641f2',  # left cam bad
#     # 'e2b845a1-e313-4a08-bc61-a5f662ed295e',  # in litpose paw list
#     # 'a8a8af78-16de-4841-ab07-fde4b5281a03',  # off by one or two, looks bad; in paw labeled data
#     # '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # off by one or two, looks bad; in paw labeled data; not in BWM dataset
#     # '0c828385-6dd6-4842-a702-c5075f5f5e81',
#     # '824cf03d-4012-4ab1-b499-c83a92c5589e',
#     # '2bdf206a-820f-402f-920a-9e86cd5388a4',
#     # '8a3a0197-b40a-449f-be55-c00b23253bbf',
#     # other issues
#     # 'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # no timestamps; not in BWM dataset
#     # '7af49c00-63dd-4fed-b2e0-1b3bd945b20b',  # right cam timestamps seem weird?
# ]


# ----------------------------
# lp model directories
# ----------------------------
# eids = [
#     # '07dc4b76-5b93-4a03-82a0-b3d9cc73f412',  # for felix
#     # '6274dda8-3a59-4aa1-95f8-a8a549c46a26',  # for felix (left timestamps bad)
#     # 'ae4a54de-43c9-4eff-8a7d-bd2d05c5f993',  # for guido
# ]
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

print('pupil pipeline finished')
for key, val in error_log.items():
    print(f'{key}: {val}\n')
