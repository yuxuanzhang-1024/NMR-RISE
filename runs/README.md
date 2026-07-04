# Runs directory

This directory stores model checkpoints and other artifacts produced or consumed by training runs. It includes NMR2Mol, MolRef, and Mol2NMR checkpoints, preprocessors, target scalers, configuration snapshots, and training logs.

Binary checkpoints and generated run artifacts are not tracked by Git. Published pretrained artifacts can be downloaded from [Napister/NMR-RISE](https://huggingface.co/Napister/NMR-RISE) using the commands in the root [README](../README.md#data-and-checkpoints). New training runs will also write their outputs here unless a different path is configured.

Typical layout:

```text
runs/
├── nmr2mol/<dataset>/...
└── mol2nmr/<dataset>/...
```
