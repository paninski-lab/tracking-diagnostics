"""File handling."""

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import os
import yaml
from typing import Dict, Union, List, Optional

from lightning_pose.utils.io import check_if_semi_supervised
import hydra


def get_model_params(cfg):
    """Returns dict containing all params considered essential for defining a model.

    Parameters
    ----------
    cfg : dict
        all relevant hparams for the given model type will be pulled from this dict

    Returns
    -------
    dict
        config dict

    """

    # start with general params
    cfg_less = {
        "train_batch_size": cfg["training"]["train_batch_size"],
        "train_prob": cfg["training"]["train_prob"],
        "train_frames": cfg["training"]["train_frames"],
        "early_stop_patience": cfg["training"]["early_stop_patience"],
        "unfreezing_epoch": cfg["training"]["unfreezing_epoch"],
        "dropout_rate": cfg["training"]["dropout_rate"],
        "max_epochs": cfg["training"]["max_epochs"],
        "rng_seed_data_pt": cfg["training"]["rng_seed_data_pt"],
        "rng_seed_model_pt": cfg["training"]["rng_seed_model_pt"],
        "downsample_factor": cfg["data"]["downsample_factor"],
        # "resnet_version": cfg["model"]["resnet_version"],
        "model_type": cfg["model"]["model_type"],
        "model_name": cfg["model"]["model_name"],
        "do_context": cfg["model"]["do_context"],
    }
    if cfg_less["model_type"] == "heatmap":
        cfg_less["heatmap_loss_type"] = cfg["model"]["heatmap_loss_type"]

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    if semi_supervised:
        cfg_less["losses_to_use"] = cfg["model"]["losses_to_use"]
        cfg_less["limit_train_batches"] = cfg["training"]["limit_train_batches"]
        if "pca_multiview" in cfg_less["losses_to_use"]:
            cfg_less["pca_multiview"] = cfg["losses"]["pca_multiview"]
        if "pca_singleview" in cfg_less["losses_to_use"]:
            cfg_less["pca_singleview"] = cfg["losses"]["pca_singleview"]
        if "temporal" in cfg_less["losses_to_use"]:
            cfg_less["temporal"] = cfg["losses"]["temporal"]
        if "unimodal_mse" in cfg_less["losses_to_use"]:
            cfg_less["unimodal_mse"] = cfg["losses"]["unimodal_mse"]
        if "unimodal_wasserstein" in cfg_less["losses_to_use"]:
            cfg_less["unimodal_wasserstein"] = cfg["losses"]["unimodal_wasserstein"]
        if "unimodal_kl" in cfg_less["losses_to_use"]:
            cfg_less["unimodal_kl"] = cfg["losses"]["unimodal_kl"]
        if "unimodal_js" in cfg_less["losses_to_use"]:
            cfg_less["unimodal_js"] = cfg["losses"]["unimodal_js"]
    else:
        cfg_less["losses_to_use"] = []

    return cfg_less


def find_model_versions(base_dir, cfg, verbose=False, keys_to_sweep=[], needs_pred_csv=True):
    """Search model versions to find if one with the same hyperparameters has been fit.

    Parameters
    ----------
    base_dir : str
        absolute path of directory containing versions
    cfg : dict
        needs to contain enough information to specify an experiment (model + training
        parameters)
    verbose : bool
        True to print desired cfg params
    keys_to_sweep : list of strs
        these can be any value
    needs_pred_csv : bool
        True to require predictions.csv file in model directory

    Returns
    -------
    list

    """

    version_dirs = collect_all_model_paths(base_dir)

    # get model-specific params
    cfg_req = get_model_params(cfg)

    # remove params if we don't want a specific value
    for key in keys_to_sweep:
        if key == "weight":
            for loss in cfg_req["losses_to_use"]:
                # delete whole loss for now, will need to update later if we want to
                # selectively sweep over loss params
                del cfg_req[loss]
        else:
            if key in cfg_req:
                del cfg_req[key]

    version_list = []
    for version_dir in version_dirs:
        # load hparams
        try:
            with open(os.path.join(version_dir, ".hydra", "config.yaml"), "rb") as f:
                cfg_ = yaml.safe_load(f)
            # collapse first level of hierarchy  # TODO: abstract w/ list comprehension
            cfg_curr = {
                **cfg_["model"],
                **cfg_["data"],
                **cfg_["losses"],
                **cfg_["training"],
            }
            if cfg_curr["losses_to_use"] is None:  # support null case in hydra
                cfg_curr["losses_to_use"] = []
            if all([cfg_curr[key] == cfg_req[key] for key in cfg_req.keys()]):
                # found match - did it finish fitting?
                if needs_pred_csv:
                    if os.path.exists(os.path.join(version_dir, "predictions.csv")):
                        version_list.append(version_dir)
                        if len(keys_to_sweep) == 0:
                            # we found the only model we're looking for
                            break
                        print(f"Found model at: {version_dir}")
                else:
                    version_list.append(version_dir)
                    if len(keys_to_sweep) == 0:
                        # we found the only model we're looking for
                        break
                    print(f"Found model at: {version_dir}")

            else:
                if verbose:
                    print(version_dir)
                    print("unmatched keys: [current vs requested]")
                    for key in cfg_req.keys():
                        if cfg_curr[key] != cfg_req[key]:
                            print(
                                "{}: {} vs {}".format(key, cfg_curr[key], cfg_req[key])
                            )
                    print()
        except FileNotFoundError:
            continue
        except KeyError:
            continue

    if verbose and len(version_list) == 0:
        print("could not find match for requested hyperparameters: {}".format(cfg_req))

    return version_list


def collect_all_model_paths(base_dir):
    """Find all subdirectories that contain a .hydra directory."""

    if not os.path.exists(base_dir):
        raise NotADirectoryError("%s is not a path" % base_dir)

    # return nothing if path is a file
    if os.path.isfile(base_dir):
        return []

    model_paths = []
    for f in os.walk(base_dir):
        if "config.yaml" in f[-1]:
            model_paths.append(os.path.dirname(f[0]))

    return model_paths


def get_best_version_from_tb_logs(version_list, metric="val_supervised_loss"):
    """Given a list of model directories, find best version based on provided metric."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob

    if len(version_list) == 0:
        return version_list

    val_losses = []
    for version in version_list:
        # assume a particular hydra output structure
        log_file = glob.glob(
            os.path.join(version, "tb_logs", "*", "*", "events.out.tfevents.*")
        )
        if len(log_file) == 0:
            # found version directory but no log file
            val_losses.append(np.inf)
            continue
        event_acc = EventAccumulator(log_file[0])
        event_acc.Reload()
        # show all tags in the log file
        # print(event_acc.Tags())

        # get wall clock, number of steps and value for a scalar defined by the `metric`
        # argument
        w_times, step_nums, vals = zip(*event_acc.Scalars(metric))
        # store best loss
        val_losses.append(np.min(vals))

    if np.min(val_losses) == np.inf:
        # return nothing if no log files found
        return []

    return version_list[np.argmin(val_losses)]

# TODO: add a function that takes in a bunch of parameters, and spits out a list of model paths that match these
""" details: you have a base config. then receive a dict with certain fields to edit
generate a few search congifs. then loop over these configs and match using the handler or find_model_versions
before everything you need to create a search dict.
 """


def create_empty_search_dict() -> Dict[str, dict]:
    keys = ["model", "data", "losses", "training"]
    search_cfg = {}
    for key in keys:
        search_cfg[key] = {}
    return search_cfg


# now need to fill in these vals
def edit_search_dict(base_cfg: Union[dict, DictConfig], **kwargs):
    # check if the value is str, float, or integer. if you're getting a list for it, it means you have to iterate
    for key, val in kwargs.items():
        print(key, val)
        if val is dict:
            for subkey, subval in val.items():
                print(subkey, subval)


def get_base_config(config_dir: str, config_name: str) -> DictConfig:
    assert(os.path.isdir(config_dir))
    hydra.initialize_config_dir(config_dir)
    cfg = hydra.compose(config_name=config_name)
    return cfg


def get_keypoint_names(csv_data: pd.DataFrame, header_rows: List[int]) -> List[str]:
    """ Get the names of the keypoints from the csv file.
    There are other ways of doing this, but this method preserves the order of the keypoints. """
    if header_rows == [0,1,2]:
        keypoint_names = [c[1] for c in csv_data.columns[1::2]]
    elif header_rows == [1,2]:
        keypoint_names = [c[0] for c in csv_data.columns[1::2]]
    else:
        raise NotImplementedError
    return keypoint_names


def update_loss_config(
        cfg: Union[dict, DictConfig],
        loss_type: str,
        loss_weight: float,
        loss_weight_dict: Optional[dict] = None):
    """Helper function to update loss config given a loss type and loss weights."""

    if loss_type == 'pca_mixed':
        cfg.model.losses_to_use = ['pca_singleview', 'pca_multiview']
        cfg.losses['pca_singleview'].log_weight = loss_weight
        cfg.losses['pca_multiview'].log_weight = loss_weight
    elif loss_type == 'single_uni':
        cfg.model.losses_to_use = ['pca_singleview', 'unimodal_mse']
        cfg.losses['pca_singleview'].log_weight = loss_weight
        cfg.losses['unimodal_mse'].log_weight = loss_weight
    elif loss_type == 'single_temp':
        cfg.model.losses_to_use = ['pca_singleview', 'temporal']
        cfg.losses['pca_singleview'].log_weight = loss_weight
        cfg.losses['temporal'].log_weight = loss_weight
    elif loss_type == 'multi_temp':
        cfg.model.losses_to_use = ['pca_multiview', 'temporal']
        cfg.losses['pca_multiview'].log_weight = loss_weight
        cfg.losses['temporal'].log_weight = loss_weight
    elif loss_type == 'single_uni_temp':
        cfg.model.losses_to_use = ['pca_singleview', 'unimodal_mse', 'temporal']
        cfg.losses['pca_singleview'].log_weight = loss_weight
        cfg.losses['unimodal_mse'].log_weight = loss_weight
        cfg.losses['temporal'].log_weight = loss_weight
    elif loss_type == 'all' or loss_type == 'combined':
        # find all losses
        if loss_weight_dict is not None:
            loss_list = []
            for loss in loss_weight_dict.keys():
                if loss == 'supervised' or loss == 'all' or loss == 'combined':
                    continue
                else:
                    loss_list.append(loss)
        else:
            loss_list = ["unimodal_mse", "temporal", "pca_multiview", "pca_singleview"]
        cfg.model.losses_to_use = loss_list
        # update weights to be the same
        for loss in cfg.model.losses_to_use:
            cfg.losses[loss].log_weight = loss_weight_dict[loss]
    else:
        cfg.model.losses_to_use = [loss_type]
        cfg.losses[loss_type].log_weight = loss_weight

    return cfg
