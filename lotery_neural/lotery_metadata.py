import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata

import numpy as np


TRACKED_PACKAGES = ["keras", "numpy", "pandas", "tensorflow"]


def configure_tensorflow_logging(verbose=False):
    if verbose:
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    try:
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
    except Exception:
        pass


def model_history_file(model_file_w_format):
    return model_file_w_format.replace('.keras', '_history.json')


def model_metadata_file(model_file_w_format):
    return model_file_w_format.replace('.keras', '_metadata.json')


def save_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_random_seed(seed, tf_verbose=False):
    configure_tensorflow_logging(tf_verbose)

    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ModuleNotFoundError:
        pass


def get_package_versions(package_names=TRACKED_PACKAGES):
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def get_final_loss(training_history):
    losses = training_history.history.get("loss", [])
    if not losses:
        return None
    return float(losses[-1])


def build_training_metadata(
    window_size,
    epochs,
    seed,
    history_size,
    temporal_pairs,
    training_pairs,
    validation_pairs,
    validation_metrics,
    final_loss,
):
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": get_package_versions(),
        "parameters": {
            "window_size": window_size,
            "epochs": epochs,
            "seed": seed,
            "validation_size": validation_pairs,
        },
        "training_games": history_size,
        "temporal_pairs": temporal_pairs,
        "training_pairs": training_pairs,
        "validation_pairs": validation_pairs,
        "final_loss": final_loss,
        "validation_metrics": validation_metrics,
    }
