import argparse
import glob
import os
import random
import pathlib
import shutil

import numpy as np
import sklearn.model_selection

from utils import get_module_logger


def clean_targets(target):
    if target.exists():
        shutil.rmtree(target)


def hardlink(files, target_path):
    for file in files:
        target = pathlib.Path(file)
        target.link_to(target_path / target.name)


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """

    files = [
        filename
        for filename in glob.glob(os.path.join(data_dir, "processed/*.tfrecord"))
    ]

    train_files, val_files = sklearn.model_selection.train_test_split(
        files, test_size=0.3
    )
    val_files, test_files = sklearn.model_selection.train_test_split(
        val_files, test_size=0.65
    )

    data_dir = pathlib.Path("dataset/splits")

    print("train_files", len(train_files))
    print("val_file", len(val_files))
    print("test_file", len(test_files))

    files_mapping = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for dir_name, files in files_mapping.items():
        data_path = data_dir / dir_name

        clean_targets(data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        hardlink(files, data_path)
    # TODO: Implement function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument("--data_dir", required=True, help="data directory")
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.data_dir)
