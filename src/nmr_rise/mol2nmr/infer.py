import os
import sys
import subprocess
import shlex
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

@hydra.main(version_base=None, config_path="../../../scripts/mol2nmr/configs", config_name="config_infer")
def main(cfg: DictConfig) -> None:
    # return to repo root so relative paths match existing scripts
    orig = get_original_cwd()
    os.chdir(orig)

    # build CLI args for the existing infer.py
    args = [sys.executable, "./src/nmr_rise/mol2nmr/inference/infer.py", "--user-dir", "./src/nmr_rise/mol2nmr", str(cfg.data_path)]

    # simple string flags
    args += ["--valid-subset", str(cfg.valid_subset)]
    args += ["--results-path", str(cfg.results_path), "--saved-dir", str(cfg.saved_dir)]
    args += ["--num-workers", str(cfg.num_workers), "--ddp-backend", str(cfg.ddp_backend)]
    args += ["--batch-size", str(cfg.batch_size), "--task", str(cfg.task)]
    args += ["--loss", str(cfg.loss), "--arch", str(cfg.arch)]
    args += ["--dict-name", str(cfg.dict_name), "--path", str(cfg.checkpoint_path)]
    if cfg.fp16:
        args.append("--fp16")
        if cfg.get("fp16_init_scale", None) is not None:
            args += ["--fp16-init-scale", str(cfg.fp16_init_scale)]
        if cfg.get("fp16_scale_window", None) is not None:
            args += ["--fp16-scale-window", str(cfg.fp16_scale_window)]
    args += ["--log-interval", str(cfg.log_interval), "--log-format", str(cfg.log_format)]
    if cfg.get("required_batch_size_multiple", None):
        args += ["--required-batch-size-multiple", str(cfg.required_batch_size_multiple)]
    # atom selection and optional flags
    if cfg.get("selected_atom"):
        args += ["--selected-atom", str(cfg.selected_atom)]
    if cfg.get("global_distance_flag"):
        args += [str(cfg.global_distance_flag)]
    if cfg.get("gaussian_kernel", False):
        args += ["--gaussian-kernel"]
    if cfg.get("atom_descriptor") is not None:
        args += ["--atom-descriptor", str(cfg.atom_descriptor)]
    if cfg.get("split_mode"):
        args += ["--split-mode", str(cfg.split_mode)]

    # allow passthrough extra args from config.extra_args (list)
    extra = cfg.get("extra_args", []) or []
    args += [str(a) for a in extra]

    # print("Running:", " ".join(shlex.quote(a) for a in args))
    # print(args)
    subprocess.run(args, check=True)

if __name__ == "__main__":
    main()