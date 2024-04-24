import argparse
from omegaconf import DictConfig
import os
import yaml

from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import predict_single_video


def run():

    args = parser.parse_args()

    cfg_file = os.path.join(args.model_dir, ".hydra", "config.yaml")
    model_cfg = DictConfig(yaml.safe_load(open(cfg_file)))
    model_cfg["data"]["csv_file"] = "CollectedData.csv"
    model_cfg["data"]["data_dir"] = args.data_dir  # for models trained on cloud
    model_cfg["data"]["video_dir"] = os.path.join(args.data_dir, "videos")
    model_cfg['dali']['base']['predict']['sequence_length'] = 384
    model_cfg['dali']['context']['predict']['sequence_length'] = 384
    model_cfg['training']['imgaug'] = 'default'

    # get model checkpoint
    ckpt_file = ckpt_path_from_base_path(args.model_dir, model_name=model_cfg.model.model_name)

    predict_single_video(
        video_file=args.video_file,
        data_module=None,
        ckpt_file=ckpt_file,
        cfg_file=model_cfg,
        preds_file=args.pred_csv_file,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # where labeled data lives
    parser.add_argument("--data_dir", type=str)

    # root model directory; should contain a .hydra subdirectory
    parser.add_argument("--model_dir", type=str)

    # absolute path to mp4
    parser.add_argument("--video_file", type=str)

    # absolute path to where predctions will be saved
    parser.add_argument("--pred_csv_file", type=str)

    run()
