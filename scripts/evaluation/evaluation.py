from nmr_rise.utils.nmr_rise import NMR_RISE, NMR_RISE_Config
from datasets import load_from_disk, Dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NMR_RISE on different datasets with 10000 samples.')
    parser.add_argument('--dataset_name', type=str, default='NMRExp', help='Name of the dataset to evaluate on.')
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size for inference.')
    parser.add_argument('--refinement_iters', type=int, default=5, help='Number of refinement iterations.')
    parser.add_argument('--modality', type=str, default='C&H', choices=['C', 'H', 'C&H'], help='Modality for inference. Options: C, H, C&H.')
    parser.add_argument('--metric_type', type=str, default='rmse', choices=['rmse', 'set_match_score'], help='Metric to evaluate.')
    parser.add_argument('--metric_ratio', nargs='+', type=float, default=[0.6, 0.1, 0.3])
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = args.dataset_name
    beam_size = args.beam_size
    modality = args.modality
    metric_type = args.metric_type
    metric_ratio = tuple(args.metric_ratio)
    refinement_iters = args.refinement_iters

    nmr_rise_config = NMR_RISE_Config()
    nmr_rise_config.nmr2mol_config['model']['model_checkpoint_path'] = f'./runs/nmr2mol/{dataset_name}/multitask_nmr2mol/version_0/checkpoints/best.ckpt'
    nmr_rise_config.nmr2mol_config['preprocessor_path'] = f'./runs/nmr2mol/{dataset_name}/multitask_nmr2mol/preprocessor.pkl'
    nmr_rise_config.nmr2mol_config['model']['batch_size'] = 64
    nmr_rise_config.molref_config['model']['model_checkpoint_path'] = f'./runs/nmr2mol/{dataset_name}/multitask_molref_10/version_0/checkpoints/best.ckpt'
    nmr_rise_config.molref_config['preprocessor_path'] = f'./runs/nmr2mol/{dataset_name}/multitask_molref_10/preprocessor.pkl'
    nmr_rise_config.molref_config['model']['batch_size'] = 64
    nmr_rise_config.mol2nmr_config['save_dir'] = f'./runs/mol2nmr/{dataset_name}/full_cc_pred_rmse_4'
    nmr_rise = NMR_RISE(nmr_rise_config)
    test_dataset = load_from_disk(f'./data/{dataset_name}/{dataset_name}-10000')
    if modality == 'C&H':
        print(f'Evaluating 10000 {dataset_name} test dataset on C&H modalities with rerank metric: {metric_type}, ratio: {tuple(metric_ratio)}, refinement iters: {refinement_iters}, beam_size: {beam_size}')
        results = nmr_rise.infer_dataset(dataset=test_dataset,
                                        show_progress=True,
                                        enable_dataset_progress=True,
                                        beam_size=beam_size,
                                        top_k=beam_size,
                                        refinement_iters=refinement_iters,
                                        rerank_nmr_metric=metric_type,
                                        rerank_metric_ratio=metric_ratio,
                                        num_proc=20)
        save_path = f'./results/{dataset_name}/10000/C&H/evaluation_{dataset_name}_10000_{beam_size}_{refinement_iters}_{metric_type}_{"".join(str(r) for r in metric_ratio)}'
        results.save_to_disk(save_path)
        print(f'Saved results to {save_path}')
    elif modality == 'C':
        print(f'Evaluating 10000 {dataset_name} test dataset on C modality with rerank metric: {metric_type}, ratio: {tuple(metric_ratio)}, refinement iters: {refinement_iters}, beam_size: {beam_size}')
        results = nmr_rise.infer_dataset(dataset=test_dataset,
                                        show_progress=True,
                                        enable_dataset_progress=True,
                                        beam_size=beam_size,
                                        top_k=beam_size,
                                        refinement_iters=refinement_iters,
                                        rerank_nmr_metric=metric_type,
                                        rerank_metric_ratio=metric_ratio,
                                        modality_to_drop='Multiplets',
                                        num_proc=20)
        save_path = f'./results/{dataset_name}/10000/C/evaluation_{dataset_name}_10000_{beam_size}_{refinement_iters}_{metric_type}_{"".join(str(r) for r in metric_ratio)}'
        results.save_to_disk(save_path)
        print(f'Saved results to {save_path}')
    elif modality == 'H':
        print(f'Evaluating 10000 {dataset_name} test dataset on H modality with rerank metric: {metric_type}, ratio: {tuple(metric_ratio)}, refinement iters: {refinement_iters}, beam_size: {beam_size}')
        results = nmr_rise.infer_dataset(dataset=test_dataset,
                                        show_progress=True,
                                        enable_dataset_progress=True,
                                        beam_size=beam_size,
                                        top_k=beam_size,
                                        refinement_iters=refinement_iters,
                                        rerank_nmr_metric=metric_type,
                                        rerank_metric_ratio=metric_ratio,
                                        modality_to_drop='Carbon',
                                        num_proc=20)
        save_path = f'./results/{dataset_name}/10000/H/evaluation_{dataset_name}_10000_{beam_size}_{refinement_iters}_{metric_type}_{"".join(str(r) for r in metric_ratio)}'
        results.save_to_disk(save_path)
        print(f'Saved results to {save_path}')
    else:
        raise ValueError(f'Invalid modality: {modality}. Options are: C, H, C&H.')

if __name__ == "__main__":
    main()