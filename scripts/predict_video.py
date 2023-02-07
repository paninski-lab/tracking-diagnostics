import argparse
from omegaconf import DictConfig
import yaml
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import predict_single_video
from lightning_pose.utils.scripts import get_imgaug_transform, get_dataset, get_data_module


def run():

    args = parser.parse_args()

    cfg_file = os.path.join(args.model_dir, ".hydra", "config.yaml")
    model_cfg = DictConfig(yaml.safe_load(open(cfg_file)))
    model_cfg["data"]["csv_file"] = "CollectedData.csv"
    model_cfg["data"]["data_dir"] = data_dir  # for models trained on cloud
    model_cfg["data"]["video_dir"] = os.path.join(data_dir, "videos")
    model_cfg['dali']['base']['predict']['sequence_length'] = 384
    model_cfg['dali']['context']['predict']['batch_size'] = 32
    model_cfg['dali']['general']['num_threads'] = 4

    # get model checkpoint
    ckpt_file = ckpt_path_from_base_path(args.model_dir, model_name=model_cfg.model.model_name)

    # build datamodule
    cfg_new = model_cfg.copy()
    cfg_new.training.train_prob = 1
    cfg_new.training.val_prob = 0
    cfg_new.training.train_frames = 1
    cfg_new.training.imgaug = 'default'
    imgaug_transform = get_imgaug_transform(cfg=cfg_new)
    dataset_new = get_dataset(
        cfg=cfg_new, data_dir=cfg_new.data.data_dir, imgaug_transform=imgaug_transform)
    datamodule_new = get_data_module(
        cfg=cfg_new, dataset=dataset_new, video_dir=cfg_new.data.video_dir)
    datamodule_new.setup()

    predict_single_video(
        video_file=args.video_file,
        data_module=datamodule_new,
        ckpt_file=ckpt_file,
        cfg_file=model_cfg,
        preds_file=args.pred_csv_file,
        gpu_id=gpu_id,
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

    # device id
    parser.add_argument("--gpu_id", type=int)

    run()
