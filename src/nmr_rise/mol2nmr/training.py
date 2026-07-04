#!/usr/bin/env python3
"""
Hydra launcher for running unicore training with parameters defined in YAML.

Usage examples:
  python hydra_train.py dataset=pretrain/full_cc_pred_rmse_5 lr=0.001 batch_size=32
  python hydra_train.py -m lr=0.005,0.001 batch_size=32,64   # multi-run sweep

This script composes a torchrun + unicore-train command from the composed Hydra config
and executes it. It uses get_original_cwd() so paths in the config can be relative to the
repository root.
"""
import os
import shutil
import subprocess
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../scripts/mol2nmr/configs", config_name="config_train")
def main(cfg: DictConfig) -> None:
    orig_cwd = get_original_cwd()

    # Resolve paths relative to repository root (where you run the script from)
    def resolve(p):
        if p is None:
            return None
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(orig_cwd, p)

    data_path = resolve(cfg.data_path)
    save_dir = resolve(cfg.save_dir)
    weight_path = resolve(cfg.weight_path)
    print(data_path, save_dir, weight_path)
    os.makedirs(save_dir, exist_ok=True)

    # Find unicore-train binary
    unicore_bin = shutil.which("unicore-train")
    if unicore_bin is None:
        raise RuntimeError("unicore-train not found in PATH. Activate your environment or install unicore.")

    # Compose optional flags
    gauss_flag = "--gaussian-kernel" if cfg.gauss_flag else ""
    global_distance_flag = cfg.global_distance_flag or ""

    finetune_flag = []
    if cfg.weight_path and cfg.weight_name:
        finetune_flag = ["--finetune-from-model", os.path.join(weight_path, f"{cfg.weight_name}.pt")]

    # Build the command list
    cmd = [
        "torchrun",
        f"--nproc_per_node={cfg.n_gpu}",
        f"--master_port={cfg.master_port}",
        unicore_bin,
        data_path,
        "--user-dir", "./src/nmr_rise/mol2nmr",
        "--train-subset", cfg.train_subset,
        "--valid-subset", cfg.valid_subset,
        "--num-workers", str(cfg.num_workers),
        "--ddp-backend", cfg.ddp_backend,
        "--task", cfg.task,
        "--loss", cfg.loss,
        "--arch", cfg.arch,
        "--optimizer", cfg.optimizer,
        "--adam-betas", str(cfg.adam_betas),
        "--adam-eps", str(cfg.adam_eps),
        "--clip-norm", str(cfg.clip_norm),
        "--lr-scheduler", cfg.lr_scheduler,
        "--lr", str(cfg.lr),
        "--warmup-ratio", str(cfg.warmup),
        "--max-epoch", str(cfg.epoch),
        "--batch-size", str(cfg.batch_size),
        "--update-freq", str(cfg.update_freq),
        "--seed", str(cfg.seed),
    ]

    # FP16 flags
    if cfg.fp16.enable:
        cmd += ["--fp16", "--fp16-init-scale", str(cfg.fp16.init_scale), "--fp16-scale-window", str(cfg.fp16.scale_window)]

    cmd += [
        "--num-classes", str(cfg.num_classes),
        "--pooler-dropout", str(cfg.dropout),
    ]

    # finetune and dict
    if finetune_flag:
        cmd += finetune_flag
    if cfg.dict_name:
        cmd += ["--dict-name", f"{cfg.dict_name}.txt"]

    cmd += [
        "--log-interval", str(cfg.log_interval),
        "--log-format", cfg.log_format,
        "--validate-interval", str(cfg.validate_interval),
        "--keep-last-epochs", str(cfg.keep_last_epochs),
        "--save-interval", str(cfg.save_interval),
        "--save-dir", save_dir,
        "--best-checkpoint-metric", cfg.best_checkpoint_metric,
        "--selected-atom", cfg.selected_atom,
        "--split-mode", cfg.split_mode,
        "--atom-descriptor", str(cfg.atom_descriptor),
    ]

    # Add optional flags that may be empty
    if global_distance_flag:
        cmd.append(global_distance_flag)
    if gauss_flag:
        cmd.append(gauss_flag)

    # For debugging: show the final composed command
    # print("Running command:")
    # print(" ".join(shlex for shlex in [str(x) for x in cmd]))

    # Prepare environment (preserve current env, but you can modify here)
    env = os.environ.copy()
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["OMP_NUM_THREADS"] = str(cfg.omp_num_threads)

    # Execute
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
