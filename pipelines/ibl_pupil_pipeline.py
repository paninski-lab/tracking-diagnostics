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
base_dir = '/media/mattw/behavior/results/pose-estimation/ibl-pupil/ensembling-expts'
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
    'make_qc_video': {
        'run': True,
        'kwargs': {
            'overwrite': False,
            'idx_beg': None,
            'idx_end': None,
            'plot_markers': True,  # only eks for now
        },
    },
    'decode': {
        'run': False,
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
        }
    },
}

# ----------------------------
# pupil eids
# ----------------------------

# litpose paper
# eids = [
#     # -------------------------
#     # original sessions
#     # -------------------------
#     # no issues
#     'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
#     '5285c561-80da-4563-8694-739da92e5dd0',
#     '8928f98a-b411-497e-aa4b-aa752434686d',
#     '3d59aa1a-b4ba-47fe-b9cf-741b5fdb0c7b',
#     '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
#     '5d6aa933-4b00-4e99-ae2d-5003657592e9',
#     'a92c4b1d-46bd-457e-a1f4-414265f0e2d4',
#     '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00',
#     '9468fa93-21ae-4984-955c-e8402e280c83',
#     '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
#     'b658bc7d-07cd-4203-8a25-7b16b549851b',
#     '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
#     '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
#     '5bcafa14-71cb-42fa-8265-ce5cda1b89e0',
#     '6364ff7f-6471-415a-ab9e-632a12052690',
#     'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
#     'd0c91c3c-8cbb-4929-8657-31f18bffc294',
#     '30e5937e-e86a-47e6-93ae-d2ae3877ff8e',
#     '931a70ae-90ee-448e-bedb-9d41f3eda647',
#     '78b4fff5-c5ec-44d9-b5f9-d59493063f00',
#     'c6db3304-c906-400c-aa0f-45dd3945b2ea',
#     '9fe512b8-92a8-4642-83b6-01158ab66c3c',
#     'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
#     'aa20388b-9ea3-4506-92f1-3c2be84b85db',
#     '0b7ee1b6-42db-46cd-a465-08f531366187',
#     'd2f5a130-b981-4546-8858-c94ae1da75ff',
#     '768a371d-7e88-47f8-bf21-4a6a6570dd6e',
#     '948fd27b-507b-41b3-bdf8-f9f5f0af8e0b',
#     '752456f3-9f47-4fbf-bd44-9d131c0f41aa',
#     '81a1dca0-cc90-47c5-afe3-c277319c47c8',
#     'a4747ac8-6a75-444f-b99b-696fff0243fd',
#     '6668c4a0-70a4-4012-a7da-709660971d7a',
#     '1ca83b26-30fc-4350-a616-c38b7d00d240',
#     '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
#     'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
#     '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
#     '3f6e25ae-c007-4dc3-aa77-450fd5705046',
#     '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
#     '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
#     'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
#     '821f1883-27f3-411d-afd3-fb8241bbc39a',
#     '75b6b132-d998-4fba-8482-961418ac957d',
#     '1b715600-0cbc-442c-bd00-5b0ac2865de1',
#     # issues
#     # '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',  # left/right timestamp issue
#     # '8207abc6-6b23-4762-92b4-82e05bed5143',  # dlc issue?
#     # 'b81e3e11-9a60-4114-b894-09f85074d9c3',  # timestamp issue
#     # '7416f387-b302-4ca3-8daf-03b585a1b7ec',  # left/right timestamp issue
#     # '952870e5-f2a7-4518-9e6d-71585460f6fe',  # left/right timestamp issue
#     #
#     # -------------------------
#     # additional repro sessions (22)
#     # -------------------------
#     # sync good
#     '034e726f-b35f-41e0-8d6c-a22cc32391fb',  # FELIX
#     '56b57c38-2699-4091-90a8-aba35103155e',  # in paw labeled data
#     'ee40aece-cffd-4edb-a4b6-155f158c666a',
#     'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
#     'dda5fc59-f09a-4256-9fb5-66c67667a466',
#     # '4b00df29-3769-43be-bb40-128b1cba6d35',  # bad video quality
#     'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
#     'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # in paw labeled data
#     '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
#     '51e53aff-1d5d-4182-a684-aba783d50ae5',
#     '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
#     '3638d102-e8b6-4230-8742-e548cd87a949',
#     '88224abb-5746-431f-9c17-17d7ef806e6a',
#     'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # in litpose paw list already
#     'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
#     '3f859b5c-e73a-4044-b49e-34bb81e96715',
#     'b22f694e-4a34-4142-ab9d-2556c3487086',
#     '746d1902-fa59-4cab-b0aa-013be36060d5',  # in litpose paw list already
#     '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
#     '0802ced5-33a3-405e-8336-b65ebc5cb07c',
#     # sync ok
#     '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # off by one; in litpose paw list already
#     'ecb5520d-1358-434c-95ec-93687ecd1396',  # off by one
#     '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # off by one?
# ]

# for guido
# eids = [
#     # 'ae4a54de-43c9-4eff-8a7d-bd2d05c5f993',
# ]

# for felix/prior paper
# eids = [
#     # initial BWM (recordings contain sensory areas VISp and LGd)
#     # # '07dc4b76-5b93-4a03-82a0-b3d9cc73f412',  # timestamp issue
#     # # '6274dda8-3a59-4aa1-95f8-a8a549c46a26',  # timestamp issue
#     # '034e726f-b35f-41e0-8d6c-a22cc32391fb',
#     # '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
#     # '09b2c4d1-058d-4c84-9fd4-97530f85baf6',
#     # '58b1e920-cfc8-467e-b28b-7654a55d0977',
#     # '0c828385-6dd6-4842-a702-c5075f5f5e81',
#     # '111c1762-7908-47e0-9f40-2f2ee55b6505',
#     # '1a507308-c63a-4e02-8f32-3239a07dc578',
#     # '3537d970-f515-4786-853f-23de525e110f',
#     # '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
#     # '41431f53-69fd-4e3b-80ce-ea62e03bf9c7',
#     # '5339812f-8b91-40ba-9d8f-a559563cc46b',
#     # '56956777-dca5-468c-87cb-78150432cc57',  # bad ROI detection
#     # '5d01d14e-aced-4465-8f8e-9a1c674f62ec',
#     # # '6713a4a7-faed-4df2-acab-ee4e63326f8d',  # timestamp issue
#     # '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
#     # '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4',
#     # '8a3a0197-b40a-449f-be55-c00b23253bbf',
#     # '90d1e82c-c96f-496c-ad4e-ee3f02067f25',
#     # '931a70ae-90ee-448e-bedb-9d41f3eda647',
#     # # 'a82800ce-f4e3-4464-9b80-4c3d6fade333',  # timestamp issues
#     # 'bda2faf5-9563-4940-a80f-ce444259e47b',
#     # 'e0928e11-2b86-4387-a203-80c77fab5d52',
#     # 'e1931de1-cf7b-49af-af33-2ade15e8abe7',
#     # # 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',  # timestamp issues
#     #
#     # SCs from karolina
#     '258b4a8b-28e3-4c18-9f86-1ea2bc0dc806',
#     # 'c7e4e6ad-280f-432f-ac85-9be299890d6e',
#     # 'd62a64f4-fdc6-448b-8f2a-53ed08d645a7',
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

# for julia/neuromodulators
# eids = [
#     '266ddf01-bc63-41b0-939c-94a543e22c37',
#     '2ffb959c-0578-40eb-9d52-294ee17c9aae',
#     '3f9cb5b5-2ca9-401d-b4c0-04358e6442b5',
#     '71e53fd1-38f2-49bb-93a1-3c826fbe7c13',
#     '8b9d37be-3974-495a-b34f-4de98569d747',
#     'cbaaf335-5fcb-4ec1-bb66-2dc9cca55ce9',
#     'd7128a09-1822-49fc-a287-870ee6f9570e',
# ]
# # hard code pupil cropping params for these eids
# PUPIL_PARAMS = {
#     '266ddf01-bc63-41b0-939c-94a543e22c37': {'width': 100, 'height': 100, 'left': 430, 'top': 270},
#     '2ffb959c-0578-40eb-9d52-294ee17c9aae': {'width': 100, 'height': 100, 'left': 440, 'top': 280},
#     '3f9cb5b5-2ca9-401d-b4c0-04358e6442b5': {'width': 100, 'height': 100, 'left': 435, 'top': 280},
#     '71e53fd1-38f2-49bb-93a1-3c826fbe7c13': {'width': 100, 'height': 100, 'left': 310, 'top': 100},
#     '8b9d37be-3974-495a-b34f-4de98569d747': {'width': 100, 'height': 100, 'left': 420, 'top': 300},
#     'cbaaf335-5fcb-4ec1-bb66-2dc9cca55ce9': {'width': 100, 'height': 100, 'left': 370, 'top': 120},
#     'd7128a09-1822-49fc-a287-870ee6f9570e': {'width': 100, 'height': 100, 'left': 420, 'top': 310},
# }


# BWM sessions
eids = [
    '6713a4a7-faed-4df2-acab-ee4e63326f8d',  # camera bumped at end
    '6ed57216-498d-48a6-b48b-a243a34710ea',
    '35ed605c-1a1a-47b1-86ff-2b56144f55af',
    'fa1f26a1-eb49-4b24-917e-19f02a18ac61',
    'ee212778-3903-4f5b-ac4b-a72f22debf03',
    '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
    '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
    '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',
    '71855308-7e54-41d7-a7a4-b042e78e3b4f',
    'f359281f-6941-4bfd-90d4-940be22ed3c3',
    '51e53aff-1d5d-4182-a684-aba783d50ae5',
    '537677fe-1e24-4755-948c-fa4a4e8ecce5',
    '5bcafa14-71cb-42fa-8265-ce5cda1b89e0',
    'd32876dd-8303-4720-8e7e-20678dc2fd71',
    '72982282-e493-45ee-87ce-aa45cb3a3ec1',
    '6434f2f5-6bce-42b8-8563-d93d493613a2',
    '64e3fb86-928c-4079-865c-b364205b502e',
    '15948667-747b-4702-9d53-354ac70e9119',
    '288bfbf3-3700-4abe-b6e4-130b5c541e61',
    'fc43390d-457e-463a-9fd4-b94a0a8b48f5',
    '93ad879a-aa42-4150-83e1-38773c9785e4',
    '3d59aa1a-b4ba-47fe-b9cf-741b5fdb0c7b',
    '8c025071-c4f3-426c-9aed-f149e8f75b7b',
    'bb8d9451-fdbd-4f46-b52e-9290e8f84d2e',
    '034e726f-b35f-41e0-8d6c-a22cc32391fb',
    'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4',
    'fa704052-147e-46f6-b190-a65b837e605e',
    '7939711b-8b4d-4251-b698-b97c1eaa846e',
    'b52182e7-39f6-4914-9717-136db589706e',
    '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e',
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
    'c99d53e6-c317-4c53-99ba-070b26673ac4',
    'd2918f52-8280-43c0-924b-029b2317e62c',
    'd839491f-55d8-4cbe-a298-7839208ba12b',
    'ecb5520d-1358-434c-95ec-93687ecd1396',
    '3663d82b-f197-4e8b-b299-7b803a155b84',
    '85dc2ebd-8aaf-46b0-9284-a197aee8b16f',
    '5386aba9-9b97-4557-abcd-abc2da66b863',
    '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4',
    '12dc8b34-b18e-4cdd-90a9-da134a9be79c',
    '0deb75fb-9088-42d9-b744-012fb8fc4afb',
    'eef82e27-c20e-48da-b4b7-c443031649e3',
    '810b1e07-009e-4ebe-930a-915e4cd8ece4',
    '0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3',
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
    'fb70ebf7-8175-42b0-9b7a-7c6e8612226e',
    '28741f91-c837-4147-939e-918d38d849f2',
    'd2f5a130-b981-4546-8858-c94ae1da75ff',
    '57b5ae8f-d446-4161-b439-b191c5e3e77b',
    'dda5fc59-f09a-4256-9fb5-66c67667a466',
    'd16a9a8d-5f42-4b49-ba58-1746f807fcc1',
    '37e96d0b-5b4b-4c6e-9b29-7edbdc94bbd0',
    '2e6e179c-fccc-4e8f-9448-ce5b6858a183',
    '1191f865-b10a-45c8-9c48-24a980fd9402',
    'f10efe41-0dc0-44d0-8f26-5ff68dca23e9',
    '6364ff7f-6471-415a-ab9e-632a12052690',
    '7be8fec4-406b-4e74-8548-d2885dcc3d5e',
    '7dd27c2f-9544-448d-918d-be0f9837b0e0',
    'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',
    '004d8fd5-41e7-4f1b-a45b-0d4ad76fe446',
    '35eeb752-8f4f-4040-9714-ba0f5b7ccdfe',
    '97c1d08d-57b4-4595-8052-91eb84ebfd74',
    'a1782f4f-86b0-480c-a7f2-3d8f1ab482ab',
    '3f71aa98-08c6-4e79-b4c8-00eae4f03eff',
    'c46b8def-620d-4d70-9397-be5694255f73',
    '15b69921-d471-4ded-8814-2adad954bcd8',
    '11163613-a6c9-4975-9586-84dc00481547',
    '5ae68c54-2897-4d3a-8120-426150704385',
    'a7eba2cf-427f-4df9-879b-e53e962eae18',
    'ebce500b-c530-47de-8cb1-963c552703ea',
    '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',
    '824cf03d-4012-4ab1-b499-c83a92c5589e',
    '8a1cf4ef-06e3-4c72-9bc7-e1baa189841b',
    '687017d4-c9fc-458f-a7d5-0979fe1a7470',
    '16693458-0801-4d35-a3f1-9115c7e5acfd',
    'e1931de1-cf7b-49af-af33-2ade15e8abe7',
    'b9c205c3-feac-485b-a89d-afc96d9cb280',
    '6cf2a88a-515b-4f7f-89a2-7d53eab9b5f4',
    '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',
    'dd87e278-999d-478b-8cbd-b5bf92b84763',
    '15f742e1-1043-45c9-9504-f1e8a53c1744',
    'a4000c2f-fa75-4b3e-8f06-a7cf599b87ad',
    'cc45c568-c3b9-4f74-836e-c87762e898c8',
    'a92c4b1d-46bd-457e-a1f4-414265f0e2d4',
    'aad23144-0e52-4eac-80c5-c4ee2decb198',
    '07dc4b76-5b93-4a03-82a0-b3d9cc73f412',
    '9468fa93-21ae-4984-955c-e8402e280c83',
    'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
    'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
    '0ac8d013-b91e-4732-bc7b-a1164ff3e445',
    '69c9a415-f7fa-4208-887b-1417c1479b48',
    'dfbe628d-365b-461c-a07f-8b9911ba83aa',
    'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
    '4503697e-af44-47d9-898d-4924be990240',
    'b22f694e-4a34-4142-ab9d-2556c3487086',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',
    'e45481fa-be22-4365-972c-e7404ed8ab5a',
    'ffef0311-8ffa-49e3-a857-b3adf6d86e12',
    '78b4fff5-c5ec-44d9-b5f9-d59493063f00',
    '196a2adf-ff83-49b2-823a-33f990049c2e',
    '6b0b5d24-bcda-4053-a59c-beaa1fe03b8f',
    '752456f3-9f47-4fbf-bd44-9d131c0f41aa',
    '3f859b5c-e73a-4044-b49e-34bb81e96715',
    'c8d46ee6-eb68-4535-8756-7c9aa32f10e4',
    '1b9e349e-93f2-41cc-a4b5-b212d7ddc8df',
    'f819d499-8bf7-4da0-a431-15377a8319d5',
    '8928f98a-b411-497e-aa4b-aa752434686d',
    '446f4724-1690-49f9-819a-2bd8e2ea88ce',
    'a2701b93-d8e1-47e9-a819-f1063046f3e7',
    '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9',
    'ee13c19e-2790-4418-97ca-48f02e8013bb',
    'db4df448-e449-4a6f-a0e7-288711e7a75a',
    '3dd347df-f14e-40d5-9ff2-9c49f84d2157',
    '3c851386-e92d-4533-8d55-89a46f0e7384',
    '158d5d35-a2ab-4a76-87b0-51048c5d5283',
    '30e5937e-e86a-47e6-93ae-d2ae3877ff8e',
    '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
    'a19c7a3a-7261-42ce-95d5-1f4ca46007ed',
    'e5c772cd-9c92-47ab-9525-d618b66a9b5d',
    '6668c4a0-70a4-4012-a7da-709660971d7a',
    '4fa70097-8101-4f10-b585-db39429c5ed0',
    '90d1e82c-c96f-496c-ad4e-ee3f02067f25',
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',
    '931a70ae-90ee-448e-bedb-9d41f3eda647',
    '02fbb6da-3034-47d6-a61b-7d06c796a830',
    '7bee9f09-a238-42cf-b499-f51f765c6ded',
    '36280321-555b-446d-9b7d-c2e17991e090',
    '741979ce-3f10-443a-8526-2275620c8473',
    'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
    '2f63c555-eb74-4d8d-ada5-5c3ecf3b46be',
    '9a629642-3a9c-42ed-b70a-532db0e86199',
    'e535fb62-e245-4a48-b119-88ce62a6fe67',
    'f25642c6-27a5-4a97-9ea0-06652db79fbd',
    '4720c98a-a305-4fba-affb-bbfa00a724a4',
    '7622da34-51b6-4661-98ae-a57d40806008',
    'b658bc7d-07cd-4203-8a25-7b16b549851b',
    'bd456d8f-d36e-434a-8051-ff3997253802',
    'ee8b36de-779f-4dea-901f-e0141c95722b',
    'b39752db-abdb-47ab-ae78-e8608bbf50ed',
    '5339812f-8b91-40ba-9d8f-a559563cc46b',
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
    'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
    'aa20388b-9ea3-4506-92f1-3c2be84b85db',
    '251ece37-7798-477c-8a06-2845d4aa270c',
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
    'e8b4fda3-7fe4-4706-8ec2-91036cfee6bd',
    'cde63527-7f5a-4cc3-8ac2-215d82e7da26',
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
    'f7335a49-4a98-46d2-a8ce-d041d2eac1d6',
    '9eec761e-9762-4897-b308-a3a08c311e69',
    '9fe512b8-92a8-4642-83b6-01158ab66c3c',
    '26aa51ff-968c-42e4-85c8-8ff47d19254d',
    'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
    '948fd27b-507b-41b3-bdf8-f9f5f0af8e0b',
    '2e22c1fc-eec6-4856-85a0-7dba8668f646',
    '64977c74-9c04-437a-9ea1-50386c4996db',
    '9a6e127b-bb07-4be2-92e2-53dd858c2762',
    '09156021-9a1d-4e1d-ae59-48cbde3c5d42',
    '90e524a2-aa63-47ce-b5b8-1b1941a1223a',
    '1d4a7bd6-296a-48b9-b20e-bd0ac80750a5',
    '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00',
    '10fac7a1-919d-4ca7-83c4-4675bd8d9416',
    '8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef',
    '68775ca0-b056-48d5-b6ae-a4c2a76ae48f',
    'f27e6cd6-cdd3-4524-b8e3-8146046e2a7d',
    'd71e565d-4ddb-42df-849e-f99cfdeced52',
    '1b61b7f2-a599-4e40-abd6-3e758d2c9e25',
    '30af8629-7b96-45b7-8778-374720ddbc5e',
    '283ecb4c-e529-409c-9f0a-8ea5191dcf50',
    '6bf810fd-fbeb-4eea-9ea7-b6791d002b22',
    'bd07e3a7-09c3-4b67-996b-42b6d5cedf1c',
    'c16d3557-b2c1-4545-93d0-112ac0915d93',
    '2038e95d-64d4-4ecb-83d0-1308d3c598f8',
    '500a71dd-8768-4211-a1fe-1fafc9fcbf29',
    '61caa69d-088b-465a-b9d0-d75341dabac6',
    '233617ec-c5cf-4eda-afc8-3b1d94f36d73',
    'ef38e503-dd79-4185-bf87-4712d4f274fe',
    'fbd28ea6-3251-48e0-b215-2c51ec5185f7',
    '9fcbd1a0-77e6-4c41-9428-eeaee74becd5',
    'c23b4118-db40-4333-af1d-933154b533c6',
    'c8e11fd8-d130-4f06-abba-9aa20240ce7c',
    '62902992-8432-46fb-af12-6392012e58c7',
    '4546cd60-fa2a-4d20-a9b6-d091e01d16f2',
    'cef05f87-161b-4031-932c-6f47daf89698',
    '75db708a-9952-4452-a5b1-a851d88f8e28',
    '7f150b7c-c261-46e6-9edb-cc391c9d9f03',
    '837b4e6a-ccfd-49b0-a1dd-3aa53fbf2ecb',
    '9fc31d79-b56f-46d0-92a0-e9563caf4a7a',
    'a34b4013-414b-42ed-9318-e93fbbc71e7b',
    '49250fba-801c-4867-a0a7-a1e19538cb61',
    '2ab7d2c2-bcb7-4ae6-9626-f3786c22d970',
    '94dabed1-741c-4ddd-a6b7-70561e27b750',
    'be164a14-6e73-42c3-ab60-d29c48693c0f',
    '37ac03f1-9831-4a30-90fc-a59e635b98bd',
    '87ad026d-5b95-4022-8d59-c260870d830f',
    '3bb54985-77b2-421d-9d1f-46185be51216',
    '6cbeead9-bb7a-40e1-8ccd-47ae60239654',
    'a2b9fbbe-79fc-4166-a16e-b307813a2f06',
    '5437ec1f-6535-470c-80b8-4c8806ee085d',
    '36573e0b-ddd6-4504-94ec-9a23c877486c',
    '04749a70-ac63-477d-8392-d4d529184fab',
    'd23b005f-46f5-4fe1-870c-f55f6eb9533d',
    '7235b10c-6621-44ea-abe9-01559633472d',
    'fd03d365-91df-41e2-ad81-9e0e4b9f5c7b',
    '62ff920c-2fdd-4feb-9d9f-0d66f2e595a1',
    '14127fdb-2e66-4823-b124-f49c128ba94d',
    '7eeb8423-49e6-4d40-ab6f-703d17af231a',
    '6ab9d98c-b1e9-4574-b8fe-b9eec88097e0',
    '37bbc17f-8dec-4fa8-97fd-e23a5c6bad1f',
    '0cf6d255-8f2f-463e-84fb-c54bacb79f51',
    'c02e5155-8e8f-427e-873d-d61490bbb9c3',
    '98e0074c-706a-40e5-bbb5-223f97585a99',
    'e6de6c35-1508-4471-b7c3-f12a5c7a6d39',
    '1dbba733-24a4-4400-9436-53f1bd8428e8',
    '2584ce3c-db10-4076-89cb-5d313138dd38',
    'e6bdb1f4-b0bf-4451-8f23-4384f2102f91',
    'f9860a11-24d3-452e-ab95-39e199f20a93',
    'c6db3304-c906-400c-aa0f-45dd3945b2ea',
    '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a',
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',
    '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
    '81a1dca0-cc90-47c5-afe3-c277319c47c8',
    '493170a6-fd94-4ee4-884f-cc018c17eeb9',
    '360eac0c-7d2d-4cc1-9dcf-79fc7afc56e7',
    'a4747ac8-6a75-444f-b99b-696fff0243fd',
    'c4432264-e1ae-446f-8a07-6280abade813',
    '9e9c6fc0-4769-4d83-9ea4-b59a1230510e',
    '09b2c4d1-058d-4c84-9fd4-97530f85baf6',
    '5d6aa933-4b00-4e99-ae2d-5003657592e9',
    '63f3dbc1-1a5f-44e5-98dd-ce25cd2b7871',
    '746d1902-fa59-4cab-b0aa-013be36060d5',
    '5b609f9b-75cb-43d3-9c39-b5b4b7a0db67',
    '56bc129c-6265-407a-a208-cc16d20a6c01',
    '1a507308-c63a-4e02-8f32-3239a07dc578',
    'f5591ac5-311d-4fa8-9bad-029d7be9c491',
    'ab8a0899-a59f-42e4-8807-95b14056104b',
    '71e03be6-b497-4991-a121-9416dcc1a6e7',
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
    '4aa1d525-5c7d-4c50-a147-ec53a9014812',
    '9b5a1754-ac99-4d53-97d3-35c2f6638507',
    '2c44a360-5a56-4971-8009-f469fb59de98',
    '3f6e25ae-c007-4dc3-aa77-450fd5705046',
    '113c5b6c-940e-4b21-b462-789b4c2be0e5',
    '0b7ee1b6-42db-46cd-a465-08f531366187',
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    'd0c91c3c-8cbb-4929-8657-31f18bffc294',
    '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
    '88224abb-5746-431f-9c17-17d7ef806e6a',
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',
    '239dd3c9-35f3-4462-95ee-91b822a22e6b',
    '9dd72e52-5393-4c08-9eca-f7dace2e59f6',
    'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
    '21e16736-fd59-44c7-b938-9b1333d25da8',
    '53738f95-bd08-4d9d-9133-483fdb19e8da',
    'd33baf74-263c-4b37-a0d0-b79dcb80a764',
    '259927fd-7563-4b03-bc5d-17b4d0fa7a55',
    '510b1a50-825d-44ce-86f6-9678f5396e02',
    '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
    'ff4187b5-4176-4e39-8894-53a24b7cf36b',
    '465c44bd-2e67-4112-977b-36e1ac7e3f8c',
    'e49d8ee7-24b9-416a-9d04-9be33b655f40',
    'ee40aece-cffd-4edb-a4b6-155f158c666a',
    'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
    'cb2ad999-a6cb-42ff-bf71-1774c57e5308',
    '934dd7a4-fbdc-459c-8830-04fe9033bc28',
    'fe1fd79f-b051-411f-a0a9-2530a02cc78d',
    '4d8c7767-981c-4347-8e5e-5d5fffe38534',
    '768a371d-7e88-47f8-bf21-4a6a6570dd6e',
    '45ef6691-7b80-4a43-bd1a-85fc00851ae8',
    'fc14c0d6-51cf-48ba-b326-56ed5a9420c3',
    '7416f387-b302-4ca3-8daf-03b585a1b7ec',
    '1928bf72-2002-46a6-8930-728420402e01',
    '22e04698-b974-4805-b241-3b547dbf37bf',
    '7cb81727-2097-4b52-b480-c89867b5b34c',
    'f84045b0-ce09-4ace-9d11-5ea491620707',
    '239cdbb1-68e2-4eb0-91d8-ae5ae4001c7a',
    '75b6b132-d998-4fba-8482-961418ac957d',
    '56b57c38-2699-4091-90a8-aba35103155e',
    '91796ceb-e314-4859-9a1f-092f85cc846a',
    '6c6983ef-7383-4989-9183-32b1a300d17a',
    '6bb5da8f-6858-4fdd-96d9-c34b3b841593',
    'f1db6257-85ef-4385-b415-2d078ec75df2',
    'd3a2b25e-46d3-4f0b-ade6-4e32255f4c35',
    '9545aa05-3945-4054-a5c3-a259f7209d61',
    '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',
    '9b528ad0-4599-4a55-9148-96cc1d93fb24',
    'b887df2c-bb9c-44c9-a9c0-538da87b2cab',
    '63c70ae8-4dfb-418b-b21b-f0b1e5fba6c9',
    '3e6a97d3-3991-49e2-b346-6948cb4580fb',
    'eacc49a9-f3a1-49f1-b87f-0972f90ee837',
    '5b49aca6-a6f4-4075-931a-617ad64c219c',
    'd901aff5-2250-467a-b4a1-0cb9729df9e2',
    '27ef44c0-acb2-4220-b776-477d0d5abd35',
    '1ca83b26-30fc-4350-a616-c38b7d00d240',
    'e0928e11-2b86-4387-a203-80c77fab5d52',
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',
    '7cc74598-9c1b-436b-84fa-0bf89f31adf6',
    '2d9bfc10-59fb-424a-b699-7c42f86c7871',
    'f4eb56a4-8bf8-4bbc-a8f3-6e6535134bad',
    '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',
    '5c0c560e-9e1f-45e9-b66e-e4ee7855be84',
    '28338153-4113-485b-835b-91cb96d984f2',
    '29a6def1-fc5c-4eea-ac48-47e9b053dcb5',
    'f56194bc-8215-4ae8-bc6a-89781ad8e050',
    '642c97ea-fe89-4ec9-8629-5e492ea4019d',
    'caa5dddc-9290-4e27-9f5e-575ba3598614',
    '5285c561-80da-4563-8694-739da92e5dd0',
    'ff96bfe1-d925-4553-94b5-bf8297adf259',
    'a9138924-4395-4981-83d1-530f6ff7c8fc',
    '0cc486c3-8c7b-494d-aa04-b70e2690bcba',
    'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
    '72028382-a869-4745-bacf-cb8789e16953',
    '821f1883-27f3-411d-afd3-fb8241bbc39a',
    '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',
    '86b6ba67-c1db-4333-add0-f8105ea6e363',
    '58b1e920-cfc8-467e-b28b-7654a55d0977',
    '08102cfc-a040-4bcf-b63c-faa0f4914a6f',
]

# repro-ephys sessions round 2
eids = [
    'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',  # Berkeley
    'e535fb62-e245-4a48-b119-88ce62a6fe67',  # Berkeley
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley
    'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley
    '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # CCU
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # CCU
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU
    '746d1902-fa59-4cab-b0aa-013be36060d5',  # CCU
    '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',  # CCU
    'ee40aece-cffd-4edb-a4b6-155f158c666a',  # CCU
    'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # CCU
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU
    'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL(C)
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL(C)
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL(C)
    '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL(C)
    'ecb5520d-1358-434c-95ec-93687ecd1396',  # CSHL(C)
    '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU
    'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
    'a8a8af78-16de-4841-ab07-fde4b5281a03',  # NYU
    '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2',  # Princeton
    'd9f0c293-df4c-410a-846d-842e47c6b502',  # Princeton
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC(H)
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC(H)  # PROBLEM
    '862ade13-53cd-4221-a3fa-dda8643641f2',  # SWC(H)  # PROBLEM
    '56b57c38-2699-4091-90a8-aba35103155e',  # SWC(M)
    '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC(M)
    '7cb81727-2097-4b52-b480-c89867b5b34c',  # SWC(M)
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # UCL
    '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL
    'b22f694e-4a34-4142-ab9d-2556c3487086',  # UCL
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',  # UCL
    'aad23144-0e52-4eac-80c5-c4ee2decb198',  # UCL
    'b196a2ad-511b-4e90-ac99-b5a29ad25c22',  # UCL
    'e45481fa-be22-4365-972c-e7404ed8ab5a',  # UCL
    'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # UCL
    'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL
    '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL
    'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA
    'dc962048-89bb-4e6a-96a9-b062a2be1426',  # UCLA
    '6899a67d-2e53-4215-a52a-c7021b5da5d4',  # UCLA
    '15b69921-d471-4ded-8814-2adad954bcd8',  # UCLA
    '5ae68c54-2897-4d3a-8120-426150704385',  # UCLA
    'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',  # UCLA
    '824cf03d-4012-4ab1-b499-c83a92c5589e',  # UCLA
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
    'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
    '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
    '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # UW
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

    print(f'eid {e + 1} / {len(eids)}: {eid}')

    try:

        pipe = PupilPipeline(
            eid=eid, one=one, base_dir=base_dir,
            allow_trial_fail=True, load_dlc=True,
            # for sessions that have no DLC traces
            # allow_trial_fail=True, load_dlc=False, pupil_crop_params=PUPIL_PARAMS[eid],
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

        # make video snippet overlaid with predicted markers
        if pipe_kwargs['make_qc_video']['run']:
            pipe.make_qc_video(**pipe_kwargs['make_qc_video']['kwargs'])

        # decode pupil diameter from simultaneously recorded neural activity
        if pipe_kwargs['decode']['run']:
            pipe.decode(**pipe_kwargs['decode']['kwargs'])

    except Exception as exception:
        error_log[eid] = exception

print('pupil pipeline finished')
for key, val in error_log.items():
    print(f'{key}: {val}\n')
