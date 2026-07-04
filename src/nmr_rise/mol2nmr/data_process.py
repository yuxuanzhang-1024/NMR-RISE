from typing import Dict, Any
import os
import logging
import datasets
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from datasets.utils import disable_progress_bar, enable_progress_bar
from nmr_rise.utils.data import (
    filter_invalid_entry,
    mol2nmr_target_generation,
    conformation_construction,
    calculate_rmse
)
import numpy as np
from nmr_rise.utils.mol2nmr import mol2nmr_inference
import shutil
logging.basicConfig(level=logging.INFO)
# get logger from the path of this file
logger = logging.getLogger(os.path.relpath(__file__))



@hydra.main(version_base=None, config_path="../../../scripts/mol2nmr/configs", config_name="config_data_process")
def main(cfg: DictConfig) -> None:
    """
    Expected config keys:
      - input_dataset_path: str
      - output_dataset_path: str
      - task_type: str, one of ['conformation_construction', 'nmr_prediction']
      - mol2nmr_ckpt: str (required for nmr_prediction)
      - batch_size: int
    Example CLI:
      python data_process.py input_dataset_path=/path/to/in output_dataset_path=/path/to/out task_type=nmr_prediction mol2nmr_ckpt=/path/to/ckpt batch_size=32
    """
    orig_cwd = get_original_cwd()

    # Resolve paths relative to repository root (where you run the script from)
    def resolve(p):
        if p is None:
            return None
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(orig_cwd, p)
    # Resolve paths relative to original working directory
    input_path = resolve(cfg.input_dataset_path)
    output_path = resolve(cfg.output_dataset_path)
    mol2nmr_ckpt_dir = (
        resolve(cfg.mol2nmr_ckpt_dir) if "mol2nmr_ckpt_dir" in cfg and cfg.mol2nmr_ckpt_dir else None
    )
    mol2nmr_ckpt_name = str(cfg.mol2nmr_ckpt_name) if "mol2nmr_ckpt_name" in cfg and cfg.mol2nmr_ckpt_name else None
    batch_size = int(cfg.get("batch_size", 32))
    task_type = str(cfg.task_type)
    show_progress = bool(cfg.get("show_progress", True))
    if show_progress:
        enable_progress_bar()
    else:
        disable_progress_bar()
    if task_type not in ["conformation_construction", "nmr_prediction"]:
        raise ValueError("task_type must be one of ['conformation_construction', 'nmr_prediction']")
    if task_type == "nmr_prediction" and (not mol2nmr_ckpt_dir or not mol2nmr_ckpt_name):
        raise ValueError("--mol2nmr_ckpt_dir and --mol2nmr_ckpt_name are required for nmr_prediction task")

    logger.info("Loading dataset from %s", input_path)
    dataset = datasets.load_from_disk(input_path)

    if task_type == "conformation_construction":
        logger.info("Constructing conformations and filtering failures...")
        for split in list(dataset.keys()):
            dataset[split] = dataset[split].map(
                conformation_construction, num_proc=os.cpu_count()
            )
            dataset[split] = dataset[split].filter(
                filter_invalid_entry, num_proc=os.cpu_count()
            )
        logger.info("Saving processed dataset to %s", output_path)
        dataset.save_to_disk(output_path)

    elif task_type == "nmr_prediction":
        logger.info("Starting NMR prediction...")
        mol2nmr_infer_args: Dict[str, Any] = {
            "save_dir": mol2nmr_ckpt_dir,
            "weight_name": mol2nmr_ckpt_name,
            "data_path": input_path,
            "batch_size": batch_size,
            "task": "mol2nmr",
            "loss": "atom_regloss_mae",
            "arch": "unimol_large",
            "dict_name": "dict",
            "selected_atom": "C&H",
            "atom_des": 0,
            "split_mode": "infer",
        }
        mol2nmr_infer = mol2nmr_inference(mol2nmr_infer_args)
        for split in list(dataset.keys()):
            logger.info("Inferring NMR data for split: %s", split)
            dataset[split] = mol2nmr_infer.infer_dataset(dataset[split], show_progress=show_progress)
            logger.info("Generating NMR targets for Mol2NMR training for split: %s", split)
            dataset[split] = dataset[split].map(mol2nmr_target_generation, num_proc=os.cpu_count())
            dataset[split] = dataset[split].map(calculate_rmse, num_proc=os.cpu_count())
            logger.info("Mean C_RMSE and H_RMSE for split %s: %.4f, %.4f", split,
                        np.mean(dataset[split]['C_rmse']), np.mean(dataset[split]['H_rmse']))
        logger.info("Saving inferred dataset to %s", output_path)
        dataset.save_to_disk(output_path)
        # copy the dict.txt file to the output_path
    if os.path.exists(os.path.join(input_path, "dict.txt")):
        shutil.copy(os.path.join(input_path, "dict.txt"), os.path.join(output_path, "dict.txt"))
        logger.info("Copied dict.txt to %s", output_path)
    else:
        logger.warning("dict.txt not found in %s", input_path)


if __name__ == "__main__":
    main()