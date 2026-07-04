"""Run configurable NMR-RISE ablation studies.

The script evaluates a Cartesian product of MolRef versions, Mol2NMR versions,
modalities, beam sizes, and refinement iteration counts. Run it from the
repository root so the default relative paths resolve correctly.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


MODALITY_DEFAULTS = {
    "C&H": {"modality_to_drop": None, "metric_ratio": (0.6, 0.3, 0.1)},
    "C": {"modality_to_drop": "Multiplets", "metric_ratio": (0.8, 0.2)},
    "H": {"modality_to_drop": "Carbon", "metric_ratio": (0.6, 0.4)},
}


@dataclass(frozen=True)
class AblationJob:
    molref_version: str
    mol2nmr_version: str
    modality: str
    beam_size: int
    refinement_iters: int
    metric_ratio: tuple[float, ...]
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NMR-RISE ablations over MolRef/Mol2NMR versions, modalities, "
            "beam sizes, and refinement iteration counts."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="NMRExp",
        choices=["USPTO-NMR", "NMRBank", "NMRExp"],
        help="Dataset-specific NMR2Mol, MolRef, and Mol2NMR checkpoints to use.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=1000,
        choices=[1000, 10000],
        help="Published evaluation subset size.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Explicit datasets.save_to_disk path; overrides automatic discovery.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples. Useful for smoke tests.",
    )
    parser.add_argument(
        "--modalities",
        "--modality",
        dest="modalities",
        nargs="+",
        default=["C&H", "C", "H"],
        choices=["C&H", "C", "H"],
        help="One or more input modalities.",
    )
    parser.add_argument(
        "--beam_sizes",
        "--beam_size",
        dest="beam_sizes",
        nargs="+",
        type=int,
        default=[10],
        help="One or more beam sizes.",
    )
    parser.add_argument(
        "--refinement_iters",
        nargs="+",
        type=int,
        default=[5],
        help="One or more iterative refinement counts.",
    )
    parser.add_argument(
        "--molref_versions",
        "--molref_version",
        dest="molref_versions",
        nargs="+",
        default=["10"],
        help="MolRef augmentation/checkpoint versions, for example: 1 3 5 10.",
    )
    parser.add_argument(
        "--mol2nmr_versions",
        "--mol2nmr_version",
        dest="mol2nmr_versions",
        nargs="+",
        default=["4"],
        help="Mol2NMR versions, for example: nmrshiftdb 0 1 2 3 4.",
    )
    parser.add_argument(
        "--metric_type",
        default="rmse",
        choices=["rmse", "set_match_score", "vec_sim"],
        help="NMR spectral similarity metric.",
    )
    parser.add_argument(
        "--metric_ratio",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional explicit metric weights. When omitted, use 0.6/0.3/0.1 "
            "for C&H, 0.8/0.2 for C, and 0.6/0.4 for H."
        ),
    )
    parser.add_argument(
        "--model_batch_size",
        type=int,
        default=64,
        help="Batch size used by NMR2Mol and MolRef inference.",
    )
    parser.add_argument(
        "--mol2nmr_batch_size",
        type=int,
        default=8,
        help="Batch size used by Mol2NMR inference.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=20,
        help="CPU process count used by dataset transformations.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root. Defaults to results/<dataset>/<size>/ablation.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip jobs whose output directory already exists.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue the grid after a model initialization or inference failure.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the job grid and resolved paths without loading models or data.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable model and datasets progress bars.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num_samples must be a positive integer.")
    if args.model_batch_size < 1 or args.mol2nmr_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers.")
    if args.num_proc < 1:
        raise ValueError("--num_proc must be a positive integer.")
    if any(value < 1 for value in args.beam_sizes):
        raise ValueError("All beam sizes must be positive integers.")
    if any(value < 0 for value in args.refinement_iters):
        raise ValueError("Refinement iteration counts cannot be negative.")

    version_pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
    for version in [*args.molref_versions, *args.mol2nmr_versions]:
        if not version_pattern.fullmatch(version):
            raise ValueError(f"Invalid checkpoint version: {version!r}")

    if args.metric_ratio is not None:
        for modality in args.modalities:
            expected_length = 3 if modality == "C&H" else 2
            if len(args.metric_ratio) != expected_length:
                raise ValueError(
                    f"{modality} requires {expected_length} metric weights, but "
                    f"--metric_ratio received {len(args.metric_ratio)}. Omit the "
                    "argument to use modality-specific defaults."
                )


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return args.dataset_path

    dataset_dirname = f"{args.dataset_name}-{args.dataset_size}"
    candidates = [
        Path("data") / args.dataset_name / dataset_dirname,
        Path("data/evaluation") / dataset_dirname,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def mol2nmr_checkpoint_dir(dataset_name: str, version: str) -> Path:
    if version in {"nmrshiftdb", "nmrshiftdb2_2024"}:
        return Path("runs/mol2nmr/nmrshiftdb2_2024")
    return Path("runs/mol2nmr") / dataset_name / f"full_cc_pred_rmse_{version}"


def lightning_checkpoint_path(run_dir: Path) -> Path:
    """Resolve a Lightning best checkpoint without assuming version_0 exists."""
    preferred = run_dir / "version_0/checkpoints/best.ckpt"
    if preferred.exists():
        return preferred

    candidates = list(run_dir.glob("version_*/checkpoints/best.ckpt"))
    if not candidates:
        return preferred

    def version_number(path: Path) -> int:
        match = re.fullmatch(r"version_(\d+)", path.parents[1].name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=version_number)


def ratio_tag(metric_ratio: Sequence[float]) -> str:
    return "".join(str(int(round(value * 10))) for value in metric_ratio)


def build_jobs(args: argparse.Namespace) -> list[AblationJob]:
    output_root = args.output_dir or (
        Path("results") / args.dataset_name / str(args.dataset_size) / "ablation"
    )
    jobs = []
    grid = itertools.product(
        args.molref_versions,
        args.mol2nmr_versions,
        args.modalities,
        args.beam_sizes,
        args.refinement_iters,
    )
    for molref_version, mol2nmr_version, modality, beam_size, refinement_iters in grid:
        metric_ratio = tuple(
            args.metric_ratio or MODALITY_DEFAULTS[modality]["metric_ratio"]
        )
        run_name = (
            f"evaluation_{args.dataset_name}_{args.dataset_size}_{beam_size}_"
            f"{refinement_iters}_{args.metric_type}_{ratio_tag(metric_ratio)}_"
            f"molref_{molref_version}_mol2nmr_{mol2nmr_version}"
        )
        output_path = (
            output_root
            / modality
            / f"molref_{molref_version}_mol2nmr_{mol2nmr_version}"
            / run_name
        )
        jobs.append(
            AblationJob(
                molref_version=molref_version,
                mol2nmr_version=mol2nmr_version,
                modality=modality,
                beam_size=beam_size,
                refinement_iters=refinement_iters,
                metric_ratio=metric_ratio,
                output_path=output_path,
            )
        )
    return jobs


def required_model_paths(args: argparse.Namespace, job: AblationJob) -> list[Path]:
    nmr2mol_dir = (
        Path("runs/nmr2mol") / args.dataset_name / "multitask_nmr2mol"
    )
    molref_dir = (
        Path("runs/nmr2mol")
        / args.dataset_name
        / f"multitask_molref_{job.molref_version}"
    )
    mol2nmr_dir = mol2nmr_checkpoint_dir(args.dataset_name, job.mol2nmr_version)
    return [
        nmr2mol_dir / "preprocessor.pkl",
        lightning_checkpoint_path(nmr2mol_dir),
        molref_dir / "preprocessor.pkl",
        lightning_checkpoint_path(molref_dir),
        mol2nmr_dir / "checkpoint_best.pt",
        mol2nmr_dir / "target_scaler.ss",
        Path("data/nmrshiftdb2_2024/dict.txt"),
        Path("data/nmrshiftdb2_2024/dataset_dict.json"),
    ]


def validate_paths(args: argparse.Namespace, job: AblationJob) -> None:
    missing = [path for path in required_model_paths(args, job) if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required checkpoint artifacts are missing:\n" + missing_list
        )


def build_model(args: argparse.Namespace, job: AblationJob):
    from nmr_rise.utils.nmr_rise import NMR_RISE, NMR_RISE_Config

    nmr2mol_dir = (
        Path("runs/nmr2mol") / args.dataset_name / "multitask_nmr2mol"
    )
    molref_dir = (
        Path("runs/nmr2mol")
        / args.dataset_name
        / f"multitask_molref_{job.molref_version}"
    )
    mol2nmr_dir = mol2nmr_checkpoint_dir(args.dataset_name, job.mol2nmr_version)

    config = NMR_RISE_Config()
    config.nmr2mol_config["model"]["model_checkpoint_path"] = str(
        lightning_checkpoint_path(nmr2mol_dir)
    )
    config.nmr2mol_config["preprocessor_path"] = str(
        nmr2mol_dir / "preprocessor.pkl"
    )
    config.nmr2mol_config["model"]["batch_size"] = args.model_batch_size

    config.molref_config["model"]["model_checkpoint_path"] = str(
        lightning_checkpoint_path(molref_dir)
    )
    config.molref_config["preprocessor_path"] = str(
        molref_dir / "preprocessor.pkl"
    )
    config.molref_config["model"]["batch_size"] = args.model_batch_size

    config.mol2nmr_config["save_dir"] = str(mol2nmr_dir)
    config.mol2nmr_config["batch_size"] = args.mol2nmr_batch_size
    return NMR_RISE(config)


def print_plan(args: argparse.Namespace, dataset_path: Path, jobs: list[AblationJob]) -> None:
    model_pairs = list(
        dict.fromkeys((job.molref_version, job.mol2nmr_version) for job in jobs)
    )
    print(f"Dataset: {dataset_path}")
    print(f"Jobs: {len(jobs)} across {len(model_pairs)} model pair(s)")
    for index, job in enumerate(jobs, start=1):
        status = " [exists]" if job.output_path.exists() else ""
        print(
            f"[{index}/{len(jobs)}] molref={job.molref_version}, "
            f"mol2nmr={job.mol2nmr_version}, modality={job.modality}, "
            f"beam={job.beam_size}, refinement_iters={job.refinement_iters}, "
            f"ratio={job.metric_ratio} -> {job.output_path}{status}"
        )


def run_jobs(args: argparse.Namespace, dataset_path: Path, jobs: list[AblationJob]) -> None:
    import torch
    from datasets import load_from_disk

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError(f"Evaluation dataset is empty: {dataset_path}")

    model_pairs = list(
        dict.fromkeys((job.molref_version, job.mol2nmr_version) for job in jobs)
    )
    completed = 0
    skipped = 0
    failed = 0

    for molref_version, mol2nmr_version in model_pairs:
        pair_jobs = [
            job
            for job in jobs
            if job.molref_version == molref_version
            and job.mol2nmr_version == mol2nmr_version
        ]
        pending_jobs = []
        for job in pair_jobs:
            if job.output_path.exists():
                if args.skip_existing:
                    print(f"Skipping existing output: {job.output_path}")
                    skipped += 1
                    continue
                raise FileExistsError(
                    f"Output already exists: {job.output_path}. Use --skip_existing "
                    "or choose a different --output_dir."
                )
            pending_jobs.append(job)
        if not pending_jobs:
            continue

        representative_job = pending_jobs[0]
        try:
            validate_paths(args, representative_job)
            print(
                f"Loading model pair: MolRef {molref_version}, "
                f"Mol2NMR {mol2nmr_version}"
            )
            model = build_model(args, representative_job)
        except Exception as error:
            failed += len(pending_jobs)
            print(
                f"Failed to initialize MolRef {molref_version} / Mol2NMR "
                f"{mol2nmr_version}: {error}"
            )
            if args.continue_on_error:
                continue
            raise

        try:
            for job in pending_jobs:
                modality_to_drop: Optional[str] = MODALITY_DEFAULTS[job.modality][
                    "modality_to_drop"
                ]
                print(
                    f"Running modality={job.modality}, beam={job.beam_size}, "
                    f"refinement_iters={job.refinement_iters}, "
                    f"metric={args.metric_type}, ratio={job.metric_ratio}"
                )
                try:
                    results = model.infer_dataset(
                        dataset=dataset,
                        show_progress=not args.no_progress,
                        enable_dataset_progress=not args.no_progress,
                        beam_size=job.beam_size,
                        top_k=job.beam_size,
                        refinement_iters=job.refinement_iters,
                        rerank_nmr_metric=args.metric_type,
                        rerank_metric_ratio=job.metric_ratio,
                        modality_to_drop=modality_to_drop,
                        num_proc=args.num_proc,
                    )
                    job.output_path.parent.mkdir(parents=True, exist_ok=True)
                    results.save_to_disk(str(job.output_path))
                    completed += 1
                    print(f"Saved results to {job.output_path}")
                except Exception as error:
                    failed += 1
                    print(f"Job failed: {error}")
                    if not args.continue_on_error:
                        raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Finished: completed={completed}, skipped={skipped}, failed={failed}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_path = resolve_dataset_path(args)
    jobs = build_jobs(args)
    print_plan(args, dataset_path, jobs)
    if args.dry_run:
        return
    run_jobs(args, dataset_path, jobs)


if __name__ == "__main__":
    main()
