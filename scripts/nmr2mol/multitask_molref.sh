#!/bin/bash
export TOKENIZERS_PARALLELISM=False

dataset_name=NMRExp
aug_scale=1
model=custom_model
lr=1e-3

pos_enc=learned
gated_linear=True
modality=multimodal

mkdir -p runs/${dataset_name}/multitask_molref_${aug_scale}/
python -m nmr_rise.nmr2mol.cli.training \
    working_dir=runs/${dataset_name}/multitask_molref_${aug_scale}/ \
    job_name=multitask_molref_${aug_scale} \
    data_path=data/${dataset_name}/revision_${aug_scale} \
    data=multimodal/multimodal_molref.yaml \
    model=${model} \
    molecules=True \
    trainer.epochs=100 \
    model.batch_size=256 \
    model.lr=${lr} \
    model.positional_encoding_type=${pos_enc} \
    model.gated_linear=${gated_linear} \
    model.optimiser=adamw \
    splitting=predefined \
    modality_dropout=[Multiplets,Carbon]

# NMRBank dataset
# mkdir -p runs/NMRBank/multitask_molref_${aug_scale}/
# python -m nmr_rise.nmr2mol.cli.training \
#     working_dir=runs/NMRBank/multitask_molref_${aug_scale}/ \
#     job_name=multitask_molref_${aug_scale} \
#     data_path=data/NMRBank/revision_${aug_scale} \
#     data=multimodal/multimodal_molref.yaml \
#     model=${model} \
#     molecules=True \
#     trainer.epochs=100 \
#     model.batch_size=256 \
#     model.lr=${lr} \
#     model.positional_encoding_type=${pos_enc} \
#     model.gated_linear=${gated_linear} \
#     model.optimiser=adamw \
#     model.model_checkpoint_path=runs/USPTO-NMR/multitask_molref_10/multitask_molref_10/version_0/checkpoints/best.ckpt \
#     preprocessor_path=runs/USPTO-NMR/multitask_molref_10/multitask_molref_10/preprocessor.pkl \
#     finetuning=True \
#     splitting=predefined \
#     modality_dropout=[Multiplets,Carbon]