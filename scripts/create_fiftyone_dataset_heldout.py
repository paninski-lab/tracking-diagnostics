"""Visualize predictions of models in a fiftyone dashboard."""

import hydra
from omegaconf import DictConfig
from lightning_pose.utils.fiftyone import (
    FiftyOneImagePlotter,
    FiftyOneKeypointVideoPlotter,
    check_dataset,
    FiftyOneFactory,
)
import fiftyone as fo

from lightning_pose.utils.scripts import pretty_print_str


@hydra.main(config_path="configs", config_name="config")
def build_fo_dataset(cfg: DictConfig) -> None:
    pretty_print_str(
        "Launching a job that creates %s FiftyOne.Dataset"
        % cfg.eval.fiftyone.dataset_to_create
    )
    FiftyOneClass = FiftyOneFactory(
        dataset_to_create=cfg.eval.fiftyone.dataset_to_create
    )()

    # update data csv
    cfg.data.csv_file = "CollectedData_new.csv"
    # point fiftyone to predicted csv files for held-out data
    csv_filename = "predictions_held-out.csv"

    fo_plotting_instance = FiftyOneClass(cfg=cfg, csv_filename=csv_filename)
    dataset = fo_plotting_instance.create_dataset()  # internally loops over models
    check_dataset(dataset)  # create metadata and print if there are problems
    fo_plotting_instance.dataset_info_print()  # print the name of the dataset

    if cfg.eval.fiftyone.launch_app_from_script:
        # launch an interactive session
        session = fo.launch_app(dataset, remote=True)
        session.wait()
    # otherwise launch from an ipython session


if __name__ == "__main__":
    build_fo_dataset()
