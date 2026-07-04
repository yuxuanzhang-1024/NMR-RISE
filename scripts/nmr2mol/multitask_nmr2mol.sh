#!/bin/bash
export TOKENIZERS_PARALLELISM=False

dataset_name=NMRExp
model=custom_model
lr=1e-3

pos_enc=learned
gated_linear=True
modality=multimodal

mkdir -p runs/${dataset_name}
python -m nmr_rise.nmr2mol.cli.training \
    working_dir=runs/${dataset_name}/ \
    job_name=multitask_nmr2mol \
    data_path=data/${dataset_name}/full \
    data=multimodal/multimodal_nmr2mol.yaml \
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
# mkdir -p runs/multitask_nmr2mol/NMRBank
# python -m nmr_rise.nmr2mol.cli.training \
#     working_dir=runs/NMRBank/ \
#     job_name=multitask_nmr2mol \
#     data_path=data/NMRBank/full \
#     data=multimodal/multimodal_nmr2mol.yaml \
#     model=${model} \
#     molecules=True \
#     trainer.epochs=100 \
#     model.batch_size=256 \
#     model.lr=${lr} \
#     model.positional_encoding_type=${pos_enc} \
#     model.gated_linear=${gated_linear} \
#     model.optimiser=adamw \
#     model.model_checkpoint_path=runs/USPTO-NMR/multitask_nmr2mol/version_0/checkpoints/best.ckpt \
#     preprocessor_path=runs/USPTO-NMR/multitask_nmr2mol/preprocessor.pkl \
#     finetuning=True \
#     splitting=predefined \
#     modality_dropout=[Multiplets,Carbon]