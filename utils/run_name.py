# utils/run_name.py
"""
Run name generation and config persistence.

Generated format:
    YYYYMMDD_<ModelClass>_<TrainerClass>_<param1><val1>_<param2><val2>_<idx>

Example:
    20260302_Net_RLTrainer_lr2p5e-04_n_epochs4_0

The trailing index auto-increments so repeated runs with the same
hyperparameters never overwrite each other's configs or weights.

Usage (from training_schedule.py):
    from utils.run_name import generate_run_name, save_config

    NAME_KEYS = ["lr", "n_epochs", "batch_size"]
    run_name = generate_run_name("Net", "RLTrainer", config, NAME_KEYS)
    save_config(run_name, config)
"""

import json
import os
from datetime import date


def _fmt(value) -> str:
    """
    Format a hyperparameter value for use in a filename.
      float 2.5e-4  ->  "2p5e-04"
      float 0.95    ->  "0p95"
      int   64      ->  "64"
    """
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def generate_run_name(
    model_class: str,
    trainer_class: str,
    config: dict,
    name_keys: list[str],
    configs_dir: str = "configs",
) -> str:
    """
    Build a unique run name and guarantee no collision with existing configs.

    Args:
        model_class:   Name of the model class (e.g. "Net")
        trainer_class: Name of the trainer class (e.g. "RLTrainer")
        config:        Full hyperparameter dict
        name_keys:     Which config keys to embed in the name
        configs_dir:   Directory to scan for existing runs (default "configs")

    Returns:
        A unique run name string.
    """
    date_str = date.today().strftime("%Y%m%d")
    param_parts = [f"{k}{_fmt(config[k])}" for k in name_keys if k in config]
    stem = "_".join([date_str, model_class, trainer_class] + param_parts)

    os.makedirs(configs_dir, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(configs_dir, f"{stem}_{idx}.json")):
        idx += 1

    return f"{stem}_{idx}"


def save_config(run_name: str, config: dict, configs_dir: str = "configs"):
    """
    Persist the full config dict as JSON under configs/<run_name>.json.

    Args:
        run_name:    The generated run name (used as the filename stem)
        config:      Hyperparameter dict to serialise
        configs_dir: Target directory (default "configs")
    """
    os.makedirs(configs_dir, exist_ok=True)
    path = os.path.join(configs_dir, f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved → {path}")
