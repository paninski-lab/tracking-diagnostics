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
├── qc_vids
└── video_preds
    ├── <eid_0>.left.rng=0.csv
    ├── ...
    └── <eid_n>.left.rng=m.csv

"""

gpu_id = 0
rng_seed = 0  # choose ensemble member for single-model analyses
tracker_name = 'heatmap_mhcrnn_tracker'
pipe_kwargs = {
    'download_data': {  # download data from flatiron
        'run': False,
        'kwargs': {}},
    'make_sync_video': {  # create sync video for manual timestamp qc
        'run': True,
        'kwargs': {
            'overwrite': False,
            'use_raw_vids': True,
            'idx_beg': None,
            'idx_end': None,
            'plot_markers': True,  # only eks for now
        },
    },
    'preprocess_video': {  # process and reencode video with ffmpeg
        'run': False,
        'kwargs': {'overwrite': False, 'mp4_file': None}},
    'reencode_video': {  # reencode already ffmpeg-processed video
        'run': False,
        'kwargs': {'overwrite': False}},
    'infer_video': {
        'run': False,
        'kwargs': {'overwrite': False, 'gpu_id': gpu_id}},
    'smooth_kalman': {
        'run': False,
        'kwargs': {'overwrite': False, 'tracker_name': tracker_name}},
    'decode': {
        'run': False,
        'kwargs': {
            # 'overwrite': False,
            'tracker_name': tracker_name,
            'trackers': ['dlc', 'lp', 'lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            # 'trackers': ['lp+ks'],  # 'dlc' | 'lp' | 'lp+ks'
            'rng_seed': rng_seed,  # for 'lp' tracker
            'date': '02-02-2023',
        },
    },
}

# ----------------------------
# paw eids: LP paper
# ----------------------------
# eids = [
#     # '032ffcdf-7692-40b3-b9ff-8def1fc18b2e',  # sync good
#     # '6ed57216-498d-48a6-b48b-a243a34710ea',  # sync good
#     # '91a3353a-2da1-420d-8c7c-fad2fedfdd18',  # sync good
#     # '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',  # sync good
#     # 'a405053a-eb13-4aa4-850c-5a337e5dc7fd',  # sync good
#     # '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # sync good
#     # '158d5d35-a2ab-4a76-87b0-51048c5d5283',  # sync good
#     # '7622da34-51b6-4661-98ae-a57d40806008',  # sync ok (off by a frame or two)
#     # 'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',  # sync good
#     # 'f9860a11-24d3-452e-ab95-39e199f20a93',  # sync good
#     # 'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # sync good
#     # '2c44a360-5a56-4971-8009-f469fb59de98',  # sync good
#     # 'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',  # sync good
#     # '27ef44c0-acb2-4220-b776-477d0d5abd35',  # sync good
#     # 'dc21e80d-97d7-44ca-a729-a8e3f9b14305',  # sync ok (off by a frame or two)
#     # '58b1e920-cfc8-467e-b28b-7654a55d0977',  # sync good
#     # 'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c',  # sync ok (off by a frame or two)
#     # '6c6b0d06-6039-4525-a74b-58cfaa1d3a60',  # sync good
#     # 'ae8787b1-4229-4d56-b0c2-566b61a25b77',  # sync good
#     # '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',  # sync good
#     # 'a7763417-e0d6-4f2a-aa55-e382fd9b5fb8',  # sync good
#     # 'ee212778-3903-4f5b-ac4b-a72f22debf03',  # sync ok (maybe off by a frame?)
#     # '64e3fb86-928c-4079-865c-b364205b502e',  # sync good
#     '034e726f-b35f-41e0-8d6c-a22cc32391fb',  # sync good
#     '6364ff7f-6471-415a-ab9e-632a12052690',  # sync ok (off by a frame or two)
#     '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',  # sync ok (off by a couple frames) [good lp+ks]
#     '15f742e1-1043-45c9-9504-f1e8a53c1744',  # sync ok (off by a couple frames) [good lp+ks]
#     'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',  # sync good
#     'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',  # sync good
#     '90e524a2-aa63-47ce-b5b8-1b1941a1223a',  # sync ok (off by a frame or two)
#     '746d1902-fa59-4cab-b0aa-013be36060d5',  # sync good
#     '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # sync good
#     '7416f387-b302-4ca3-8daf-03b585a1b7ec',  # sync good
#     '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',  # sync good
#     'd901aff5-2250-467a-b4a1-0cb9729df9e2',  # sync good
#     '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',  # sync good
#     'd2832a38-27f6-452d-91d6-af72d794136c',  # sync good
#     'ee40aece-cffd-4edb-a4b6-155f158c666a',  # sync good
#     'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # sync good
#     'dda5fc59-f09a-4256-9fb5-66c67667a466',  # sync good
#     'ecb5520d-1358-434c-95ec-93687ecd1396',  # sync ok (off by one)
#     '4b00df29-3769-43be-bb40-128b1cba6d35',  # sync good
#     '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # sync ok (off by one?)
#     'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # sync good
# ]

# eids = [
#     # -------------------------
#     # original sessions
#     # -------------------------
#     # sync good
#     '032ffcdf-7692-40b3-b9ff-8def1fc18b2e',
#     '6ed57216-498d-48a6-b48b-a243a34710ea',
#     '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
#     '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',
#     'a405053a-eb13-4aa4-850c-5a337e5dc7fd',
#     '1b715600-0cbc-442c-bd00-5b0ac2865de1',
#     '158d5d35-a2ab-4a76-87b0-51048c5d5283',
#     'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
#     'f9860a11-24d3-452e-ab95-39e199f20a93',
#     'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
#     '2c44a360-5a56-4971-8009-f469fb59de98',
#     'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
#     '27ef44c0-acb2-4220-b776-477d0d5abd35',
#     '58b1e920-cfc8-467e-b28b-7654a55d0977',
#     '6c6b0d06-6039-4525-a74b-58cfaa1d3a60',
#     'ae8787b1-4229-4d56-b0c2-566b61a25b77',
#     '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',
#     'a7763417-e0d6-4f2a-aa55-e382fd9b5fb8',
#     '64e3fb86-928c-4079-865c-b364205b502e',
#     '034e726f-b35f-41e0-8d6c-a22cc32391fb',
#     'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
#     'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
#     '746d1902-fa59-4cab-b0aa-013be36060d5',
#     '7416f387-b302-4ca3-8daf-03b585a1b7ec',
#     '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',
#     'd901aff5-2250-467a-b4a1-0cb9729df9e2',
#     '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',
#     'd2832a38-27f6-452d-91d6-af72d794136c',
#     # sync ok
#     '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # off by a frame
#     '7622da34-51b6-4661-98ae-a57d40806008',  # off by a frame or two
#     'dc21e80d-97d7-44ca-a729-a8e3f9b14305',  # off by a frame or two
#     'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c',  # off by a frame or two
#     'ee212778-3903-4f5b-ac4b-a72f22debf03',  # maybe off by a frame?
#     '6364ff7f-6471-415a-ab9e-632a12052690',  # off by a frame or two
#     '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',  # off by a couple frames [good lp+ks]
#     '15f742e1-1043-45c9-9504-f1e8a53c1744',  # off by a couple frames [good lp+ks]
#     '90e524a2-aa63-47ce-b5b8-1b1941a1223a',  # off by a frame or two
#     # sync BAD
#     # '03d9a098-07bf-4765-88b7-85f8d8f620cc',
#     # 'e2b845a1-e313-4a08-bc61-a5f662ed295e',
#     # '8d316998-28c3-4265-b029-e2ca82375b2f',
#     # '30af8629-7b96-45b7-8778-374720ddbc5e',
#     # '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',  # off by a couple frames
#     # 'bb099402-fb31-4cfd-824e-1c97530a0875',
#     # 'e349a2e7-50a3-47ca-bc45-20d1899854ec',  # off by a couple frames [good lp+ks]
#     # '5adab0b7-dfd0-467d-b09d-43cb7ca5d59c',  # off by a couple frames [eh lp+ks]
#     # '1ec23a70-b94b-4e9c-a0df-8c2151da3761',  # off by a couple frames [good lp+ks]
#     # other issues
#     # '1e45d992-c356-40e1-9be1-a506d944896f',  # sync good; timestamps bad
#     # '66d98e6e-bcd9-4e78-8fbb-636f7e808b29',  # off by a frame or two [bad lp+ks]
#     #
#     # -------------------------
#     # additional repro sessions (19)
#     # -------------------------
#     # sync good
#     'ee40aece-cffd-4edb-a4b6-155f158c666a',
#     'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
#     'dda5fc59-f09a-4256-9fb5-66c67667a466',
#     '4b00df29-3769-43be-bb40-128b1cba6d35',
#     'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
#     '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
#     '51e53aff-1d5d-4182-a684-aba783d50ae5',
#     '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
#     '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # in litpose pupil list already
#     '3638d102-e8b6-4230-8742-e548cd87a949',
#     '88224abb-5746-431f-9c17-17d7ef806e6a',
#     'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
#     '3f859b5c-e73a-4044-b49e-34bb81e96715',
#     'b22f694e-4a34-4142-ab9d-2556c3487086',
#     '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
#     '0802ced5-33a3-405e-8336-b65ebc5cb07c',
#     # sync ok
#     'ecb5520d-1358-434c-95ec-93687ecd1396',  # off by one
#     '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # off by one?
#     '8928f98a-b411-497e-aa4b-aa752434686d',  # off by one; in litpose pupil list already
# ]

# --------------------------------------------------------------
# for repro-ephys paper: round 1
# --------------------------------------------------------------
# eids = [
#     # sync good
#     '56b57c38-2699-4091-90a8-aba35103155e',  # in paw labeled data
#     '41872d7f-75cb-4445-bb1a-132b354c44f0',  # not in BWM dataset
#     'ee40aece-cffd-4edb-a4b6-155f158c666a',
#     '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # not in BWM dataset
#     'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
#     'dda5fc59-f09a-4256-9fb5-66c67667a466',
#     '4b00df29-3769-43be-bb40-128b1cba6d35',
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

eids = [
    # FINAL LIST
    # '56b57c38-2699-4091-90a8-aba35103155e',  # SWC
    # '41872d7f-75cb-4445-bb1a-132b354c44f0',  # SWC
    # 'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC
    # '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC
    # '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # CCU (0.95/.16/.13)
    # 'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # CCU (1.58/.06/.06)
    # 'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU (1.33/.14/.26)
    # '746d1902-fa59-4cab-b0aa-013be36060d5',  # CCU (1.14/.11/.07)
    # '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU (0.21/.14/.16)
    # '0802ced5-33a3-405e-8336-b65ebc5cb07c',  # CCU (0.36/.11/.09)
    # 'ee40aece-cffd-4edb-a4b6-155f158c666a',  # CCU (0.56/.08/.24)
    # 'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL
    # 'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL
    # '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL
    # 'ecb5520d-1358-434c-95ec-93687ecd1396',  # CSHL; off by one
    # # '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL; bad pupil video quality; sync off by 1
    # 'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
    # 'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
    # '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',  # Berkeley
    # '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley; off by one?
    # # 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley; sync off by 1-2
    # '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU
    # 'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
    # # '754b74d5-7a06-4004-ae0c-72a10b6ed2e6',  # NYU; bad pupil video quality
    # # '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU; sync off by 1-2; video pos bad
    # '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # UCL (1.02/.15/.15)
    # '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL (1.55/.07/.17)
    # 'b22f694e-4a34-4142-ab9d-2556c3487086',  # UCL (0.52/.11/.13)
    # '0a018f12-ee06-4b11-97aa-bbbff5448e9f',  # UCL (0.42/.17/.14)
    # 'aad23144-0e52-4eac-80c5-c4ee2decb198',  # UCL (0.42/.01/.12)
    # # '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL; off by one
    # NEW
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU; off by one frame, ok?
    'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC
    'b196a2ad-511b-4e90-ac99-b5a29ad25c22',  # UCL
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2',  # Princeton
    'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA
    'e45481fa-be22-4365-972c-e7404ed8ab5a',  # UCL
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
    '7cb81727-2097-4b52-b480-c89867b5b34c',  # SWC
    'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL
    'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL; sync off by one frame, ok?
    'd9f0c293-df4c-410a-846d-842e47c6b502',  # Princeton
    '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # UCL
]


# --------------------------------------------------------------
# for repro-ephys paper: round 2
# --------------------------------------------------------------
eids = [
    'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
    # 'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
    # '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',  # Berkeley
    # 'e535fb62-e245-4a48-b119-88ce62a6fe67',  # Berkeley
    # '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley
    # 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley
    # '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # CCU
    # 'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # CCU
    # 'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU
    # '746d1902-fa59-4cab-b0aa-013be36060d5',  # CCU
    # '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU
    # '0802ced5-33a3-405e-8336-b65ebc5cb07c',  # CCU
    # 'ee40aece-cffd-4edb-a4b6-155f158c666a',  # CCU
    # 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # CCU
    # '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU
    # 'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL(C)
    # '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL(C)
    # 'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL(C)
    # '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL(C)
    # 'ecb5520d-1358-434c-95ec-93687ecd1396',  # CSHL(C)
    # '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU
    # 'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
    # 'a8a8af78-16de-4841-ab07-fde4b5281a03',  # NYU
    # '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU
    # '73918ae1-e4fd-4c18-b132-00cb555b1ad2',  # Princeton
    # 'd9f0c293-df4c-410a-846d-842e47c6b502',  # Princeton
    # 'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC(H)
    # '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC(H)
    # '862ade13-53cd-4221-a3fa-dda8643641f2',  # SWC(H)
    # '56b57c38-2699-4091-90a8-aba35103155e',  # SWC(M)
    # '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC(M)
    # '7cb81727-2097-4b52-b480-c89867b5b34c',  # SWC(M)
    # '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # UCL
    # '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL
    # 'b22f694e-4a34-4142-ab9d-2556c3487086',  # UCL
    # '0a018f12-ee06-4b11-97aa-bbbff5448e9f',  # UCL
    # 'aad23144-0e52-4eac-80c5-c4ee2decb198',  # UCL
    # 'b196a2ad-511b-4e90-ac99-b5a29ad25c22',  # UCL
    # 'e45481fa-be22-4365-972c-e7404ed8ab5a',  # UCL
    # 'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL
    # '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # UCL
    # 'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL
    # '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL
    # 'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA
    # 'dc962048-89bb-4e6a-96a9-b062a2be1426',  # UCLA
    # '6899a67d-2e53-4215-a52a-c7021b5da5d4',  # UCLA
    # '15b69921-d471-4ded-8814-2adad954bcd8',  # UCLA
    # '5ae68c54-2897-4d3a-8120-426150704385',  # UCLA
    # 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',  # UCLA
    # '824cf03d-4012-4ab1-b499-c83a92c5589e',  # UCLA
    # '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
    # 'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
    # '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
    # '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # UW
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

one = ONE()

# ----------------------------
# run single-view pipeline
# ----------------------------
if pipe_kwargs['download_data']['run'] \
        or pipe_kwargs['preprocess_video']['run'] \
        or pipe_kwargs['reencode_video']['run'] \
        or pipe_kwargs['infer_video']['run']:
    views = ['right', 'left']
    error_log_single = {}
    for e, eid in enumerate(eids):
        for view in views:
            print(f'eid {e}: {eid} ({view})')
            try:

                pipe = PawPipeline(eid=eid, one=one, view=view, base_dir=base_dir)
                print(pipe.paths.alyx_session_path)

                # download data from flatiron
                if pipe_kwargs['download_data']['run']:
                    pipe.download_data(**pipe_kwargs['download_data']['kwargs'])

                # preprocess video and re-encode into yuv420p format need for lightning pose
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

            except Exception as exception:
                print(exception)
                error_log_single[f'{eid}_{view}'] = exception

    print('paw single-view pipeline finished')
    for key, val in error_log_single.items():
        print(f'{key}: {val}\n')

# ----------------------------
# run multi-view pipeline
# ----------------------------
if pipe_kwargs['make_sync_video']['run'] \
        or pipe_kwargs['smooth_kalman']['run']:
    error_log_multi = {}
    for e, eid in enumerate(eids):
        print(f'eid {e}: {eid}')
        try:

            pipe = MultiviewPawPipeline(eid=eid, one=one, base_dir=base_dir)
            print(pipe.paths.alyx_session_path)

            # create sync video for manual timestamp qc
            if pipe_kwargs['make_sync_video']['run']:
                pipe.make_sync_video(**pipe_kwargs['make_sync_video']['kwargs'])

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
            print(exception)
            error_log_multi[eid] = exception

    print('paw multi-view pipeline finished')
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
        try:
            pipe = PawPipeline(eid=eid, one=one, view='left', base_dir=base_dir)
            print(pipe.paths.alyx_session_path)
            if pipe_kwargs['decode']['run']:
                pipe.decode(paw='paw_l', **pipe_kwargs['decode']['kwargs'])
        except Exception as exception:
            error_log_decode[f'{eid}_left'] = exception

        # decode right paw from right view (paw_r, note feature name is not switched - on purpose)
        try:
            pipe = PawPipeline(eid=eid, one=one, view='right', base_dir=base_dir)
            print(pipe.paths.alyx_session_path)
            if pipe_kwargs['decode']['run']:
                pipe.decode(paw='paw_l', **pipe_kwargs['decode']['kwargs'])
        except Exception as exception:
            error_log_decode[f'{eid}_left'] = exception

    print('paw decoding pipeline finished')
    for key, val in error_log_decode.items():
        print(f'{key}: {val}\n')
