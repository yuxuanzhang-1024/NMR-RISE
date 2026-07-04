# Results directory

This directory stores generated inference and evaluation outputs, such as ranked molecular candidates, iterative refinement histories, metric summaries, ablation results, and prepared LLM reranking requests.

Generated results are not tracked by Git because they can be large and are reproducible from the published datasets and checkpoints. The root [README](../README.md#usage) describes end-to-end inference, while [scripts/README.md](../scripts/README.md) documents evaluation and paper-reproduction commands.

For example, the README inference workflow saves its output to `results/my_inference/`.
