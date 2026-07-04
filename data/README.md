# Data directory

This directory stores downloaded and processed datasets used by NMR-RISE, including training splits, evaluation subsets, calibration sets, and optional LLM reranking demonstrations.

The dataset files are not tracked by Git because the complete collection is large. Download the required files from [Napister/NMR-RISE](https://huggingface.co/Napister/NMR-RISE) by following the commands in the root [README](../README.md#data-and-checkpoints).

Typical contents include:

```text
data/
├── NMRExp/
├── NMRBank/
├── USPTO-NMR/
├── evaluation/
└── nmrshiftdb2_2024/
```

`data/NMRExp/LLM_Rerank/` contains the curated NMRExp cases used by the optional LLM reranking workflow.
