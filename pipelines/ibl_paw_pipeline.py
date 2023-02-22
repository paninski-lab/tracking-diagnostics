"""Pipeline function that will perform the following steps of the ibl paw pipeline:

1. re-encode mp4 videos using yuv420
2. run inference on videos
3. smooth predictions using ensemble kalman smoother
4. decode paw speed from neural activity

"""

from one.api import ONE
import os

from diagnostics.paper_ibl import PawPipeline, MultiviewPawPipeline

# ----------------------------
# pipeline user options
# ----------------------------
data_dir = '/media/mattw/behavior/pose-estimation-data-final/ibl-paw'
base_dir = '/media/mattw/behavior/results/pose-estimation/ibl-paw/ensembling-expts'
"""directory structure

base_dir
├── decoding
├── figs
├── kalman_outputs
├── models
├── paw_preds
│   ├── paw.<eid_0>.left.rng=0.csv
│   ├── ...
│   └── paw.<eid_n>.left.rng=m.csv
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
    'smooth_kalman': {
        'run': False, 'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'decode': {
        'run': True,
        'kwargs': {
            # 'overwrite': False,
            'tracker_name': tracker_name,
            'trackers': ['dlc', 'lp', 'lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            # 'trackers': ['lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            'rng_seed': rng_seed,  # for 'lp' tracker
            'date': '02-02-2023',
        }},
}

# ----------------------------
# paw eids
# ----------------------------
eids = [
    # new (fail/total)
    '032ffcdf-7692-40b3-b9ff-8def1fc18b2e',  # sync good
    '6ed57216-498d-48a6-b48b-a243a34710ea',  # sync good
    '91a3353a-2da1-420d-8c7c-fad2fedfdd18',  # sync good
    '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',  # sync good
    'a405053a-eb13-4aa4-850c-5a337e5dc7fd',  # sync good
    # '03d9a098-07bf-4765-88b7-85f8d8f620cc',  # SYNC BAD
    # 'e2b845a1-e313-4a08-bc61-a5f662ed295e',  # SYNC BAD
    # '8d316998-28c3-4265-b029-e2ca82375b2f',  # SYNC BAD
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # sync good
    '158d5d35-a2ab-4a76-87b0-51048c5d5283',  # sync good
    '7622da34-51b6-4661-98ae-a57d40806008',  # sync ok (off by a frame or two)
    'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',  # sync good
    # '30af8629-7b96-45b7-8778-374720ddbc5e',  # SYNC BAD
    '66d98e6e-bcd9-4e78-8fbb-636f7e808b29',  # sync ok (off by a frame or two)
    'f9860a11-24d3-452e-ab95-39e199f20a93',  # sync good
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # sync good
    '2c44a360-5a56-4971-8009-f469fb59de98',  # sync good
    'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',  # sync good
    '27ef44c0-acb2-4220-b776-477d0d5abd35',  # sync ???? (no mvmt in vid)
    'dc21e80d-97d7-44ca-a729-a8e3f9b14305',  # sync ok (off by a frame or two)
    '58b1e920-cfc8-467e-b28b-7654a55d0977',  # sync good
    'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c',  # sync ???? (no mvmt in vid)
    # random
    '6c6b0d06-6039-4525-a74b-58cfaa1d3a60',  # sync good
    'ae8787b1-4229-4d56-b0c2-566b61a25b77',  # sync good
    '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',  # sync good
    # '1ec23a70-b94b-4e9c-a0df-8c2151da3761',  # SYNC BAD
    'a7763417-e0d6-4f2a-aa55-e382fd9b5fb8',  # sync good
    # '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',  # SYNC BAD
    # 'bb099402-fb31-4cfd-824e-1c97530a0875',  # SYNC BAD
    # '1e45d992-c356-40e1-9be1-a506d944896f',  # sync good; timestamps bad?
    'ee212778-3903-4f5b-ac4b-a72f22debf03',  # sync good
    '64e3fb86-928c-4079-865c-b364205b502e',  # sync good
    '034e726f-b35f-41e0-8d6c-a22cc32391fb',  # sync good
    '6364ff7f-6471-415a-ab9e-632a12052690',  # sync good
    '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',  # sync good
    '15f742e1-1043-45c9-9504-f1e8a53c1744',  # sync ok (off by a frame or two)
    'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',  # sync good
    'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',  # sync good
    '90e524a2-aa63-47ce-b5b8-1b1941a1223a',  # sync good
    '746d1902-fa59-4cab-b0aa-013be36060d5',  # sync good
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # sync good
    '7416f387-b302-4ca3-8daf-03b585a1b7ec',  # sync good
    '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',  # sync good
    'd901aff5-2250-467a-b4a1-0cb9729df9e2',  # sync good
    '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',  # sync good
    'd2832a38-27f6-452d-91d6-af72d794136c',  # sync good
    'e349a2e7-50a3-47ca-bc45-20d1899854ec',  # sync ok (off by a frame or two)
    '5adab0b7-dfd0-467d-b09d-43cb7ca5d59c',  # sync good
]

# ----------------------------
# lp model directories
# ----------------------------
model_dirs = {
    '0': os.path.join(base_dir, 'models/discerning-hertz-4605-exp0/outputs/2023-02-06/19-40-18'),
    '1': os.path.join(base_dir, 'models/discerning-hertz-4605-exp1/outputs/2023-02-06/19-40-16'),
    '2': os.path.join(base_dir, 'models/discerning-hertz-4605-exp2/outputs/2023-02-06/19-40-14'),
    '3': os.path.join(base_dir, 'models/discerning-hertz-4605-exp3/outputs/2023-02-06/20-06-43'),
    '4': os.path.join(base_dir, 'models/discerning-hertz-4605-exp4/outputs/2023-02-06/19-40-14'),
}

# ----------------------------
# run single-view pipeline
# ----------------------------
one = ONE()

if pipe_kwargs['reencode_video']['run'] or pipe_kwargs['infer_video']['run']:
    views = ['right', 'left']
    error_log_single = {}
    for e, eid in enumerate(eids):
        for view in views:
            pipe = PawPipeline(eid=eid, one=one, view=view, base_dir=base_dir)
            print(f'eid {e}: {eid} ({view})')
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
            except Exception as exception:
                error_log_single[f'{eid}_{view}'] = exception

    print('single-view pipeline finished')
    for key, val in error_log_single.items():
        print(f'{key}: {val}\n')

# ----------------------------
# run multi-view pipeline
# ----------------------------
if pipe_kwargs['smooth_kalman']['run']:
    error_log_multi = {}
    for e, eid in enumerate(eids):
        pipe = MultiviewPawPipeline(eid=eid, one=one, base_dir=base_dir)
        print(f'eid {e}: {eid}')
        print(pipe.paths.alyx_session_path)
        try:
            # smooth results using ensemble kalman smoother
            if pipe_kwargs['smooth_kalman']['run']:
                pipe.smooth_kalman(
                    preds_csv_files={
                        'left': pipe.kalman_markers_file(view='left'),
                        'right': pipe.kalman_markers_file(view='right'),
                    },
                    model_dirs=model_dirs,
                    **pipe_kwargs['smooth_kalman']['kwargs']
                )
        except Exception as exception:
            error_log_multi[eid] = exception

    print('multi-view pipeline finished')
    for key, val in error_log_multi.items():
        print(f'{key}: {val}\n')

# ----------------------------
# run decoding pipeline
# ----------------------------
if pipe_kwargs['decode']['run']:
    error_log_decode = {}
    for e, eid in enumerate(eids):
        print(f'eid {e}: {eid}')

        # decode left paw from left view (paw_r)
        # pipe = PawPipeline(eid=eid, one=one, view='left', base_dir=base_dir)
        # print(pipe.paths.alyx_session_path)
        # try:
        #     if pipe_kwargs['decode']['run']:
        #         pipe.decode(paw='paw_r', **pipe_kwargs['decode']['kwargs'])
        # except Exception as exception:
        #     error_log_decode[f'{eid}_left'] = exception

        # decode right paw from right view (paw_r, note feature name is not switched - on purpose)
        pipe = PawPipeline(eid=eid, one=one, view='right', base_dir=base_dir)
        print(pipe.paths.alyx_session_path)
        # try:
        if pipe_kwargs['decode']['run']:
            pipe.decode(paw='paw_r', **pipe_kwargs['decode']['kwargs'])
        # except Exception as exception:
        #     error_log_decode[f'{eid}_left'] = exception

    print('decoding pipeline finished')
    for key, val in error_log_decode.items():
        print(f'{key}: {val}\n')
