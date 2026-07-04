import contextlib
import json
import logging
import os
import pickle
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List
from typing import Optional
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import numpy as np
from nmr_rise.nmr2mol.cli.utils import StreamToLogger  # type: ignore
from nmr_rise.nmr2mol.configuration import DEFAULT_SETTINGS
from nmr_rise.nmr2mol.data.data_utils import load_preprocessors
from nmr_rise.nmr2mol.data.datamodules import MultiModalDataModule
from nmr_rise.nmr2mol.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from nmr_rise.nmr2mol.data.augmentations import augment
from nmr_rise.nmr2mol.modeling.wrapper import HFWrapper
# from analytical_fm.trainer.trainer import build_trainer
from nmr_rise.nmr2mol.utils import (
    calc_sampling_metrics,
    calculate_training_steps,
    fail_safe_conditional_distributed_barrier,
    seed_everything,
)
from datasets import load_from_disk, DatasetDict, Dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_tensors_device(obj, device='cuda'):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: convert_tensors_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_device(item, device) for item in obj)
    else:
        # For other types, return as-is
        return obj

def update_config(config: DictConfig, config_dict: Dict[str, Any]) -> DictConfig:

    # Update config with provided settings recursively
    for key, value in config_dict.items():
        if isinstance(value, dict) and key in config:
            config[key] = update_config(config[key], value)
        else:
            config[key] = value

    return config

class nmr2mol_inference:
    def __init__(self, config_name, model_config_name, data_config_name, config_dict):
        with hydra.initialize(config_path='../../../scripts/nmr2mol/configs', version_base = None):
            config = hydra.compose(config_name=config_name)

        with hydra.initialize(config_path='../../../scripts/nmr2mol/configs/model', version_base = None):
            model_config = hydra.compose(config_name=model_config_name)

        with hydra.initialize(config_path='../../../scripts/nmr2mol/configs/data/multimodal', version_base = None):
            data_config = hydra.compose(config_name=data_config_name)

        config.model = model_config
        config.data = data_config
        config = update_config(config, config_dict)

        seed_everything()
        self.data_config = OmegaConf.to_container(config["data"].copy(), resolve=True)
        self.model_config = OmegaConf.to_container(config["model"].copy(), resolve=True) # type: ignore
        self.config = OmegaConf.to_container(config, resolve=True)
        if self.config["preprocessor_path"] is None:
            preprocessor_path = (Path(self.config["working_dir"]) / self.config["job_name"] / "preprocessor.pkl")
        else:
            preprocessor_path = Path(self.config["preprocessor_path"])
        self.data_config, self.preprocessors = pd.read_pickle(preprocessor_path)
        best_checkpoint = torch.load(self.model_config['model_checkpoint_path'])

        self.model = HFWrapper(
            data_config=self.data_config,
            target_tokenizer=self.preprocessors['Smiles'],
            modality_dropout=self.config['modality_dropout'],
            **self.model_config,
        )

        self.model.load_state_dict(best_checkpoint["state_dict"])
        self.model.eval()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.rename_columns = dict()
        for modality in data_config.keys():
            if isinstance(data_config[modality]["column"], str):
                if data_config[modality]["column"] not in ["percentage"] and ("alignment" not in data_config[modality] or not data_config[modality]["alignment"]):
                    self.rename_columns[data_config[modality]["column"]] = modality
        
    def get_data_loader(self, dataset : Dataset):
        dataset_dict = DatasetDict({"test": dataset})
        processed_dataset_dict = DatasetDict()
        for dataset_key in dataset_dict.keys():
            selected_dataset = dataset_dict[dataset_key]
            processed_dataset = selected_dataset.rename_columns(self.rename_columns)
            processed_dataset_dict[dataset_key] = processed_dataset
        print(processed_dataset_dict)
        data_module = MultiModalDataModule(
                dataset=processed_dataset_dict,
                preprocessors=self.preprocessors,
                data_config=self.data_config,
                model_type=self.model_config["model_type"],
                batch_size=self.model_config["batch_size"],
                num_workers=self.config['num_cpu'],
                extra_columns=[self.config["predict_class"]]
            )

        data_loader = data_module.inference_dataloader(dataset = processed_dataset_dict['test'])
        return data_loader

    def infer_dataset(self, 
                      dataset_path : str = None, 
                      dataset : Dataset = None, 
                      split : str = None, 
                      beam_size : int = 10, 
                      modality_to_drop : Optional[str] = None,
                      show_progress : bool = False):
        if dataset_path is not None:
            if split is None:
                raise ValueError("When providing dataset_path, split must be specified.")
            dataset_dict = load_from_disk(dataset_path)
            if split not in dataset_dict:
                raise ValueError(f"Split '{split}' not found in the dataset at {dataset_path}. Available splits: {list(dataset_dict.keys())}")
            dataset = dataset_dict[split]
        elif dataset is not None:
            dataset_dict = DatasetDict({"test": dataset})
        else:
            raise ValueError("Either dataset_path or dataset must be provided.")
        
        if split is not None and split not in ['train', 'valid', 'test']:
            raise ValueError("split variable must be one of 'train', 'valid', or 'test'.")
        # dataset_dict['train'] = augment(dataset_dict['train'], self.config["augment"], self.config["num_cpu"])
        processed_dataset_dict = DatasetDict()
        for dataset_key in dataset_dict.keys():
            selected_dataset = dataset_dict[dataset_key]
            processed_dataset = selected_dataset.rename_columns(self.rename_columns)
            processed_dataset_dict[dataset_key] = processed_dataset

        data_module = MultiModalDataModule(
                dataset=processed_dataset_dict,
                preprocessors=self.preprocessors,
                data_config=self.data_config,
                model_type=self.model_config["model_type"],
                batch_size=self.model_config["batch_size"],
                num_workers=self.config['num_cpu'],
                extra_columns=[self.config["predict_class"]]
            )
        dataset = processed_dataset_dict[split] if split is not None else processed_dataset_dict['test']
        data_loader = data_module.inference_dataloader(dataset = dataset)

        results = []
        for batch in tqdm(data_loader, disable=not show_progress):
            batch = convert_tensors_device(batch, device=self.model.device)
            output, scores = self.model.generate(batch, n_beams=beam_size, modality_to_drop=modality_to_drop, return_scores=True)
            # convert the idx to token
            output = self.preprocessors['Smiles'].batch_decode(output, skip_special_tokens=True)
            scores = torch.exp(scores).cpu().numpy().tolist()
            output = [s.replace(' ', '') for s in output]
            for i in range(len(batch['target_smiles'])):
                results.append({'true': batch['target_smiles'][i], 'pred': output[i*beam_size:(i+1)*beam_size], 'scores': scores[i*beam_size:(i+1)*beam_size]})

        return results
    
    def infer_single_entry(self, nmr_data:dict, beam_size = 10, show_progress = False):
        # check if nmr_data keys are in self.rename_columns values
        dataset = Dataset.from_list([nmr_data])
        results = self.infer_dataset(dataset=dataset, beam_size=beam_size, show_progress = show_progress)
        return results
    
    def infer_batch_entry(self, nmr_data_list:List[dict], beam_size = 10, show_progress = False):
        dataset = Dataset.from_list(nmr_data_list)
        results = self.infer_dataset(dataset=dataset, beam_size=beam_size, show_progress = show_progress)
        return results
