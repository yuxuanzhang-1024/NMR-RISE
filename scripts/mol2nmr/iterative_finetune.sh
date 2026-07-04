#!/bin/bash
dataset_name=NMRBank
master_port=33332
iters=4
infer_batch_size=128
train_batch_size=32
export CUDA_VISIBLE_DEVICES=0

# Iteratively finetune the mol2nmr model on the constructed dataset
for i in $(seq 0 $iters)
do
    if [ "$i" -eq 0 ]; then
        input_dataset_path=./data/${dataset_name}/mol2nmr/full_cc
        output_dataset_path=./data/${dataset_name}/mol2nmr/full_cc_pred_rmse_0
        mol2nmr_ckpt_dir=./runs/mol2nmr/nmrshiftdb2_2024
        mol2nmr_ckpt_name=checkpoint_best
    else
        input_dataset_path=./data/${dataset_name}/mol2nmr/full_cc_pred_rmse_$((i-1))
        output_dataset_path=./data/${dataset_name}/mol2nmr/full_cc_pred_rmse_${i}
        mol2nmr_ckpt_dir=./runs/mol2nmr/${dataset_name}/full_cc_pred_rmse_$((i-1))
        mol2nmr_ckpt_name=checkpoint_best
    fi
    python -m nmr_rise.mol2nmr.data_process \
        input_dataset_path=${input_dataset_path} \
        output_dataset_path=${output_dataset_path} \
        task_type=nmr_prediction \
        mol2nmr_ckpt_dir=${mol2nmr_ckpt_dir} \
        mol2nmr_ckpt_name=${mol2nmr_ckpt_name} \
        batch_size=${infer_batch_size} \
        show_progress=True

    save_dir=./runs/mol2nmr/${dataset_name}/full_cc_pred_rmse_${i}
    data_path=./data/${dataset_name}/mol2nmr/full_cc_pred_rmse_${i}
    if [ -d "${save_dir}" ]; then
        rm -rf ${save_dir}
        echo "Folder remove at: ${save_dir}"
    fi
    mkdir -p ${save_dir}
    echo "Folder created at: ${save_dir}"
    python -m nmr_rise.mol2nmr.training \
        master_port=${master_port} \
        data_path=${data_path} \
        save_dir=${save_dir} \
        weight_path=./runs/mol2nmr/weights \
        weight_name=pretraining_molecular \
        batch_size=${train_batch_size} \
        epoch=100 | tee ./runs/mol2nmr/${dataset_name}/full_cc_pred_rmse_${i}/training.log

    python -m nmr_rise.mol2nmr.infer \
        data_path=${data_path} \
        saved_dir=${save_dir} \
        results_path=${save_dir} | tee ${save_dir}/infer.log
done
