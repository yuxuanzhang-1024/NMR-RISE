# Model Training and Paper Reproduction

This guide contains the training and experiment-reproduction workflows for NMR-RISE. Complete the environment setup and artifact downloads described in the [main README](../README.md) first, and run all commands below from the repository root.

## Model training

Paper-scale training requires substantial GPU memory and storage. The manuscript used 100 epochs and batch size 256 for NMR2Mol/MolRef, and 100 epochs and batch size 32 for Mol2NMR fine-tuning.

### 1. NMR2Mol

Set `dataset_name` in `scripts/nmr2mol/multitask_nmr2mol.sh`, ensure `data/<dataset>/full` exists, and run:

```bash
bash scripts/nmr2mol/multitask_nmr2mol.sh
```

To keep outputs compatible with the inference and evaluation paths, set the launcher's `working_dir` to `runs/nmr2mol/<dataset>/`; the current shell script defaults to `runs/<dataset>/`.

For NMRBank, the manuscript initializes from the USPTO-NMR checkpoint and fine-tunes on NMRBank. The commented transfer-learning block in the script shows the required `model_checkpoint_path`, `preprocessor_path`, and `finetuning=True` overrides.

### 2. MolRef

MolRef is trained on observed spectra paired with NMR2Mol candidates. Generate the candidate-pair datasets with the corresponding notebook:

- `notebooks/data_process/USPTO-NMR-790K.ipynb`
- `notebooks/data_process/NMRBank.ipynb`
- `notebooks/data_process/NMRExp.ipynb`

These notebooks construct `revision_1`, `revision_3`, `revision_5`, and `revision_10` datasets. Then set `dataset_name` and `aug_scale` in the launcher and run:

```bash
bash scripts/nmr2mol/multitask_molref.sh
```

The main experiments use the `multitask_molref_10` checkpoint. Set `working_dir` so the resulting checkpoint is stored under `runs/nmr2mol/<dataset>/multitask_molref_10/`, which is the path consumed by evaluation. NMRBank again uses transfer learning from USPTO-NMR.

### 3. Mol2NMR iterative fine-tuning

Mol2NMR starts from the `nmrshiftdb2_2024` checkpoint. Each round predicts atom-level shifts, matches predictions to experimental peaks to create pseudo-labels, and fine-tunes on the updated dataset.

Prepare `data/<dataset>/mol2nmr/full_cc` with the dataset notebook, set `dataset_name`, GPU, batch sizes, and iteration count in the script, then run:

```bash
bash scripts/mol2nmr/iterative_finetune.sh
```

The paper configuration performs five rounds (`0` through `4`) and uses `full_cc_pred_rmse_4` for final inference.

## Reproducing the paper experiments

### 1. Download artifacts

For each dataset, download the 10K evaluation split and the three model checkpoints as described in [Data and checkpoints](../README.md#data-and-checkpoints). Create the expected links:

```bash
for DATASET in USPTO-NMR NMRBank NMRExp; do
  mkdir -p "data/${DATASET}"
  ln -sfn "../evaluation/${DATASET}-10000" "data/${DATASET}/${DATASET}-10000"
done
```

### 2. Main 10K evaluation

`scripts/evaluation/evaluation.py` reproduces the iterative expert-model evaluation. Examples:

```bash
# Combined 13C + 1H
python scripts/evaluation/evaluation.py \
  --dataset_name NMRExp \
  --modality 'C&H' \
  --beam_size 10 \
  --refinement_iters 5 \
  --metric_type rmse \
  --metric_ratio 0.6 0.3 0.1

# 13C only
python scripts/evaluation/evaluation.py \
  --dataset_name NMRExp \
  --modality C \
  --beam_size 10 \
  --refinement_iters 5 \
  --metric_type rmse \
  --metric_ratio 0.8 0.2

# 1H only
python scripts/evaluation/evaluation.py \
  --dataset_name NMRExp \
  --modality H \
  --beam_size 10 \
  --refinement_iters 5 \
  --metric_type rmse \
  --metric_ratio 0.6 0.4
```

Repeat with `NMRBank` and `USPTO-NMR` for Table 2 and the single-modality supplementary results.

### 3. Ablations

All four expert-model ablation studies use `scripts/evaluation/evaluation_ablation.py`. The examples below use the NMRExp-1K calibration subset. Change `--dataset_size` to `10000` for the full benchmark.

| Study | Variable being ablated | Fixed controls in the example |
|---|---|---|
| Metric | NSSS type and CNSSS/HNSSS/OCS weights | MolRef 10, Mol2NMR 4, beam 10, five refinements, C+H input |
| Parameters | Beam size and refinement iterations | MolRef 10, Mol2NMR 4, manuscript metric weights |
| Mol2NMR | Initial model and iterative fine-tuning rounds 0-4 | MolRef 10, beam 10, five refinements |
| MolRef | Training augmentation scales 1, 3, 5, and 10 | Mol2NMR 4, beam 10, five refinements |

Before starting a large grid, append `--dry_run` to inspect the number of jobs and output paths. Results are saved under `results/NMRExp/1000/ablation/`; version and parameter values are encoded in every output directory name.

#### 3.1 Metric and weight ablation

Keep the model versions and inference parameters fixed while sweeping normalized RMSE versus set-match score and the C+H metric weights. The following reproduces the C+H ratios used by the metric study:

```bash
METRIC_RATIOS=(
  "0.8 0.1 0.1"
  "0.7 0.2 0.1"
  "0.6 0.3 0.1"
  "0.7 0.1 0.2"
  "0.6 0.1 0.3"
  "0.3 0.1 0.6"
  "0.2 0.1 0.7"
  "0.8 0.2 0.0"
  "0.6 0.4 0.0"
  "0.4 0.6 0.0"
  "0.2 0.8 0.0"
  "1.0 0.0 0.0"
  "0.0 1.0 0.0"
  "0.0 0.0 1.0"
)

for METRIC_TYPE in rmse set_match_score; do
  for METRIC_RATIO in "${METRIC_RATIOS[@]}"; do
    read -r -a RATIO <<< "${METRIC_RATIO}"
    python scripts/evaluation/evaluation_ablation.py \
      --dataset_name NMRExp \
      --dataset_size 1000 \
      --molref_version 10 \
      --mol2nmr_version 4 \
      --modalities 'C&H' \
      --beam_size 10 \
      --refinement_iters 5 \
      --metric_type "${METRIC_TYPE}" \
      --metric_ratio "${RATIO[@]}" \
      --skip_existing
  done
done
```

For C-only or H-only studies, pass a two-value ratio and select one modality. For example, use `--modalities C --metric_ratio 0.8 0.2` or `--modalities H --metric_ratio 0.6 0.4`.

#### 3.2 Inference-parameter ablation

Vary beam size and the number of iterative refinements while keeping both model versions fixed. This command creates 45 jobs: 3 modalities × 3 beam sizes × 5 refinement settings.

```bash
python scripts/evaluation/evaluation_ablation.py \
  --dataset_name NMRExp \
  --dataset_size 1000 \
  --molref_version 10 \
  --mol2nmr_version 4 \
  --modalities 'C&H' C H \
  --beam_size 10 15 20 \
  --refinement_iters 1 2 3 4 5 \
  --metric_type rmse \
  --skip_existing \
  --continue_on_error
```

Because `--metric_ratio` is omitted, the script automatically uses `(0.6, 0.3, 0.1)` for C+H, `(0.8, 0.2)` for C, and `(0.6, 0.4)` for H.

#### 3.3 Mol2NMR-version ablation

Compare the original NMRShiftDB2 model, denoted by `nmrshiftdb`, with iterative fine-tuning rounds `0` through `4`. MolRef remains fixed at augmentation scale 10.

```bash
python scripts/evaluation/evaluation_ablation.py \
  --dataset_name NMRExp \
  --dataset_size 1000 \
  --molref_version 10 \
  --mol2nmr_version nmrshiftdb 0 1 2 3 4 \
  --modalities 'C&H' C H \
  --beam_size 10 \
  --refinement_iters 5 \
  --metric_type rmse \
  --skip_existing \
  --continue_on_error
```

#### 3.4 MolRef-version ablation

Compare MolRef checkpoints trained with candidate augmentation scales 1, 3, 5, and 10. Mol2NMR remains fixed at the final iterative model, version 4.

```bash
python scripts/evaluation/evaluation_ablation.py \
  --dataset_name NMRExp \
  --dataset_size 1000 \
  --molref_version 1 3 5 10 \
  --mol2nmr_version 4 \
  --modalities 'C&H' C H \
  --beam_size 10 \
  --refinement_iters 5 \
  --metric_type rmse \
  --skip_existing \
  --continue_on_error
```

For a quick end-to-end check before any full experiment, add `--num_samples 10`. Do not use `--num_samples` when producing the final ablation tables.

Published intermediate `results/` directories can be downloaded from Hugging Face to verify analysis without rerunning the GPU-intensive inference:

```bash
hf download Napister/NMR-RISE \
  --repo-type model \
  --include "results/NMRExp/**" \
  --local-dir .
```

Use `notebooks/result_analysis/analysis.ipynb` to regenerate summary tables and plots. Reference CSV tables are under `notebooks/result_analysis/tables/`.

### 4. Reproducibility checklist

- Use the published dataset split and matching dataset-specific checkpoints.
- Use beam size 10, top-k 10, and five refinement rounds for the main experiments.
- Use normalized RMSE and the modality-specific weights in [Recommended scoring settings](../README.md#recommended-scoring-settings).
- Run from the repository root so Hydra and relative paths resolve correctly.
- Record GPU model, CUDA/PyTorch versions, random seeds, and any batch-size changes.
- Distinguish the 1K metric-calibration subsets, 10K main benchmark subsets, and targeted 200-case LLM subsets.

[Back to the main README](../README.md)
