from nmr_rise.utils.nmr2mol import nmr2mol_inference
from nmr_rise.utils.mol2nmr import mol2nmr_inference
from datasets import load_from_disk, DatasetDict, Dataset
import os
from datasets.utils import disable_progress_bar, enable_progress_bar
from nmr_rise.utils.data import conformation_construction, peak_match, hnmr_list_generation, set2vec, euclidean_distance, set_match_score_weighted
import logging
import numpy as np
import warnings
from typing import Optional
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
warnings.filterwarnings("ignore")

class NMR_RISE_Config:
    def __init__(self):
        self.nmr2mol_config = {
            'working_dir': './runs/',
            'molecules': True,
            'trainer': {'epochs': 100},
            'model': {'batch_size': 32,
                    'lr': 0.001, 
                    'positional_encoding_type': 'learned', 
                    'gated_linear': True, 
                    'optimiser': 'adamw',
                    'model_checkpoint_path': ''},
            'modality_dropout':['Multiplets', 'Carbon'],
            'preprocessor_path': '',
        }
        self.molref_config = {
            'working_dir': './runs/',
            'molecules': True,
            'trainer': {'epochs': 100},
            'model': {'batch_size': 32,
                    'lr': 0.001, 
                    'positional_encoding_type': 'learned', 
                    'gated_linear': True, 
                    'optimiser': 'adamw',
                    'model_checkpoint_path': ''},
            'modality_dropout':['Multiplets', 'Carbon'],
            'preprocessor_path': '',
        }
        self.mol2nmr_config = {
            'save_dir': '',
            'weight_name': 'checkpoint_best',
            'data_path': './data/nmrshiftdb2_2024',
            'batch_size': 8,
            'task': 'mol2nmr',
            'loss': 'atom_regloss_mae',
            'arch': 'unimol_large',
            'dict_name': 'dict',
            'selected_atom': 'C&H',
            'atom_des': 0,
            'split_mode': 'infer'}

def calculate_rmse(matched_data):
    """Calculate RMSE between experimental and predicted chemical shifts"""
    squared_diff = []
    for v in matched_data.values():
        squared_diff.append((v['exp_peak'] - v['pred_peak'])**2)
    rmse = np.sqrt(np.mean(squared_diff))
    return rmse

def calculate_metric_entry(entry, col_name='candidates', modality_to_drop: Optional[str] = None):
    """Calculate RMSE for a batch of NMR data and SMILES predictions"""
    # HNMR
    if modality_to_drop == 'Carbon' or modality_to_drop is None:
        hnmr_list = hnmr_list_generation(entry['h_nmr_peaks'])
        for i in range(len(entry[col_name])):
            atoms = entry[col_name][i]['atoms']
            pred_nmr = entry[col_name][i]['predict']
            hnmr_pred = [pred_nmr[i] for i in range(len(atoms)) if atoms[i] == 'H']
            hnmr_match = peak_match(hnmr_list, hnmr_pred)
            hnmr_rmse = calculate_rmse(hnmr_match)
            hnmr_vec_sim = euclidean_distance(set2vec(np.array(hnmr_list), nmr_type='H'), set2vec(np.array(sorted(hnmr_pred)), nmr_type='H'))
            hnmr_set_match_score = set_match_score_weighted(np.array(hnmr_list), np.array(hnmr_pred), sigma=1)
            entry[col_name][i]['hnmr_rmse'] = hnmr_rmse
            entry[col_name][i]['hnmr_vec_sim'] = hnmr_vec_sim
            entry[col_name][i]['hnmr_set_match_score'] = hnmr_set_match_score
    # CNMR
    if modality_to_drop == 'Multiplets' or modality_to_drop is None:
        cnmr_list = [cnmr['delta (ppm)'] for cnmr in entry['c_nmr_peaks']]
        for i in range(len(entry[col_name])):
            atoms = entry[col_name][i]['atoms']
            pred_nmr = entry[col_name][i]['predict']
            cnmr_pred = [pred_nmr[i] for i in range(len(atoms)) if atoms[i] == 'C']
            cnmr_match = peak_match(cnmr_list, cnmr_pred)
            cnmr_rmse = calculate_rmse(cnmr_match)
            cnmr_vec_sim = euclidean_distance(set2vec(np.array(cnmr_list), nmr_type='C'), set2vec(np.array(sorted(cnmr_pred)), nmr_type='C'))
            cnmr_set_match_score = set_match_score_weighted(np.array(cnmr_list), np.array(cnmr_pred), sigma=10)
            entry[col_name][i]['cnmr_rmse'] = cnmr_rmse
            entry[col_name][i]['cnmr_vec_sim'] = cnmr_vec_sim
            entry[col_name][i]['cnmr_set_match_score'] = cnmr_set_match_score

    return entry

def normalize_rmse_metric(entry, col_name='candidates'):
    rmse_edge_mapping = {'cnmr_rmse': 5, 'hnmr_rmse': 0.5}
    for key in ['cnmr_rmse', 'hnmr_rmse']:
        for i in range(len(entry[col_name])):
            entry[col_name][i][f'{key}_normalized'] = np.exp(-entry[col_name][i][key] / rmse_edge_mapping[key])
    return entry

def candidates_info_record(entry, iter_num: int):
    candidate_info = []
    if entry['candidates'] is not None:
        for candidate in entry['candidates']:
            try:
                pred_hnmr = [candidate['predict'][i] for i in range(len(candidate['predict'])) if candidate['atoms'][i] == 'H']
                pred_cnmr = [candidate['predict'][i] for i in range(len(candidate['predict'])) if candidate['atoms'][i] == 'C']
                candidate_info.append({'smiles': candidate['smiles'],
                                        'pred_hnmr': pred_hnmr,
                                        'pred_cnmr': pred_cnmr,
                                        'hnmr_rmse': candidate['hnmr_rmse'] if 'hnmr_rmse' in candidate else None,
                                        'cnmr_rmse': candidate['cnmr_rmse'] if 'cnmr_rmse' in candidate else None,
                                        'hnmr_rmse_normalized': candidate['hnmr_rmse_normalized'] if 'hnmr_rmse_normalized' in candidate else None,
                                        'cnmr_rmse_normalized': candidate['cnmr_rmse_normalized'] if 'cnmr_rmse_normalized' in candidate else None,
                                        'hnmr_vec_sim': candidate['hnmr_vec_sim'] if 'hnmr_vec_sim' in candidate else None,
                                        'cnmr_vec_sim': candidate['cnmr_vec_sim'] if 'cnmr_vec_sim' in candidate else None,
                                        'hnmr_set_match_score': candidate['hnmr_set_match_score'] if 'hnmr_set_match_score' in candidate else None,
                                        'cnmr_set_match_score': candidate['cnmr_set_match_score'] if 'cnmr_set_match_score' in candidate else None,
                                        'type': candidate['type'],
                                        'source': candidate['source'],
                                        'refined_smiles': candidate['refined_smiles'] if 'refined_smiles' in candidate else None,
                                        'refinement_score': candidate['refinement_score'] if 'refinement_score' in candidate else None,
                                        'molref_score': candidate['molref_score'] if 'molref_score' in candidate else None,
                                        })
            except:
                continue
    entry['candidates_info_{}'.format(iter_num)] = candidate_info if candidate_info else None
    return entry

def sort_candidates(entry, 
                    col_name='candidates',
                    rerank_nmr_metric: str = 'rmse', 
                    rerank_metric_ratio: set = (1.0, 0.0, 0.0), 
                    top_k: int = 10, 
                    modality_to_drop: Optional[str] = None):
    rerank_metric_col_mapping = {'rmse': {'cnmr': 'cnmr_rmse_normalized', 'hnmr': 'hnmr_rmse_normalized'},
                                'vec_sim': {'cnmr': 'cnmr_vec_sim', 'hnmr': 'hnmr_vec_sim'},
                                'set_match_score': {'cnmr': 'cnmr_set_match_score', 'hnmr': 'hnmr_set_match_score'}}
    
    if rerank_nmr_metric not in rerank_metric_col_mapping.keys():
        raise ValueError("Invalid rerank_nmr_metric. Choose from 'rmse', 'vec_sim', 'set_match_score'.")
    def rerank_metric_cal(x):
        cnmr_sim_metric = rerank_metric_col_mapping[rerank_nmr_metric]['cnmr']
        hnmr_sim_metric = rerank_metric_col_mapping[rerank_nmr_metric]['hnmr']
        if rerank_nmr_metric == 'rmse':
            if modality_to_drop == 'Carbon':
                metric_score = rerank_metric_ratio[0]*x[hnmr_sim_metric]+rerank_metric_ratio[1]*x['molref_score']
            elif modality_to_drop == 'Multiplets':
                metric_score = rerank_metric_ratio[0]*x[cnmr_sim_metric]+rerank_metric_ratio[1]*x['molref_score']
            else:
                metric_score = rerank_metric_ratio[0]*x[cnmr_sim_metric]+rerank_metric_ratio[1]*x[hnmr_sim_metric]+rerank_metric_ratio[2]*x['molref_score']
        else:
            if modality_to_drop == 'Carbon':
                metric_score = rerank_metric_ratio[0]*x[hnmr_sim_metric]+rerank_metric_ratio[1]*x['molref_score']
            elif modality_to_drop == 'Multiplets':
                metric_score = rerank_metric_ratio[0]*x[cnmr_sim_metric]+rerank_metric_ratio[1]*x['molref_score']
            else:
                metric_score = rerank_metric_ratio[0]*x[cnmr_sim_metric]+rerank_metric_ratio[1]*x[hnmr_sim_metric]+rerank_metric_ratio[2]*x['molref_score']
        return np.round(metric_score, 4)

    candidates = entry[col_name]
    candidates = sorted(candidates, key=rerank_metric_cal, reverse=True)
    candidates_top_k = candidates[:top_k]
    for c in candidates[top_k:]:
        if c['smiles'] not in entry['history_valid_mols']:
            entry['history_valid_mols'].append(c['smiles'])
    entry[col_name] = candidates_top_k
    return entry

class NMR_RISE:
    def __init__(self, nmr_rise_config: NMR_RISE_Config = NMR_RISE_Config()):
        self.nmr2mol_config = nmr_rise_config.nmr2mol_config
        self.molref_config = nmr_rise_config.molref_config
        self.mol2nmr_config = nmr_rise_config.mol2nmr_config
        self.nmr2mol_infer = nmr2mol_inference(config_name='config_train', 
                                        model_config_name='custom_model',
                                        data_config_name='multimodal_nmr2mol',
                                        config_dict = self.nmr2mol_config)
        
        self.molref_infer = nmr2mol_inference(config_name='config_train', 
                                        model_config_name='custom_model',
                                        data_config_name='multimodal_molref',
                                        config_dict = self.molref_config)

        self.mol2nmr_infer = mol2nmr_inference(self.mol2nmr_config)

    def candidate_nmr_gen(self, candidate_list: list[dict], show_progress: bool = False, num_proc: int = None):
        candidate_dataset = Dataset.from_list(candidate_list)

        # Candidate molecule conformation construction and filtering invalid molecules
        candidate_cc_dataset = candidate_dataset.map(lambda entry: conformation_construction(entry), num_proc=os.cpu_count() if num_proc is None else num_proc)
        candidate_cc_dataset = candidate_cc_dataset.filter(lambda x: x['is_converted'])
        logging.info(f"Stage 2/3: Ratio of valid candidate molecules for conformation construction: {len(candidate_cc_dataset)}/{len(candidate_dataset)}")
        if len(candidate_cc_dataset) == 0:
            return []
        
        # Generate NMR predictions for candidate molecules
        candidate_with_nmr = self.mol2nmr_infer.infer_dataset(candidate_cc_dataset, show_progress=show_progress)
        return candidate_with_nmr

    def candidate_pool_update(self, 
                        dataset: Dataset, 
                        candidate_list: list[dict], 
                        top_k: int = 10, 
                        beam_size: int = 10,
                        show_progress: bool = False, 
                        rerank_nmr_metric: str = 'rmse',
                        rerank_metric_ratio: set = (1.0, 0.0, 0.0),
                        iter_num: int = 1,
                        modality_to_drop: Optional[str] = None,
                        num_proc: int = None,
                        return_initial_refinement: bool = False,
                        precomputed_items: dict = None):
        logging.info(f"Stage 3: Refinement Round {iter_num}---Candidate Pool Update---Generate NMR Predictions for New Candidates.")
        if precomputed_items is None:
            # Generate NMR predictions for initial candidate molecules
            candidate_with_nmr = self.candidate_nmr_gen(candidate_list, show_progress=show_progress, num_proc=num_proc)

            # Build candidate assignment dictionary
            candidate_assign_dict = {}
            idx_entry_mapping = {entry['dataset_idx']: entry for entry in dataset}
            unmatched_count = 0
            for i in range(len(candidate_with_nmr)):
                idx = candidate_with_nmr[i]['dataset_idx']
                target_molecule_formula = idx_entry_mapping[idx]['molecular_formula']
                candidate_molecule_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(candidate_with_nmr[i]['smiles']))
                if target_molecule_formula != candidate_molecule_formula:
                    unmatched_count += 1
                    continue
                else:
                    if idx not in candidate_assign_dict:
                        candidate_assign_dict[idx] = []
                    candidate_assign_dict[idx].append({k: v for k, v in candidate_with_nmr[i].items() if k in ['smiles', 'type', 'source', 'num', 'predict', 'atoms']})
            logging.info(f"Stage 3: Refinement Round {iter_num}---Candidate Pool Update---Ratio of candidate molecules with matched molecular formula: {len(candidate_with_nmr) - unmatched_count}/{len(candidate_with_nmr)}")

            # For first iteration, filter entries without valid candidates and valid refined candidates
            if iter_num == 1:
                filter_idx = []
                for i in range(len(dataset)):
                    if dataset[i]['candidates'] is None:
                        if i not in candidate_assign_dict.keys():
                            filter_idx.append(i)
                dataset = dataset.filter(lambda x, idx: idx not in filter_idx, with_indices=True)
                if filter_idx:
                    logging.info(f"Entry {filter_idx} has no valid candidates and is excluded.")
                def initial_candidate_update(entry):
                    if entry['candidates'] is None:
                        return {**entry, 'candidates': candidate_assign_dict[entry['dataset_idx']]}
                    if entry['dataset_idx'] in candidate_assign_dict:
                        return {**entry, 'candidates': entry['candidates'] + candidate_assign_dict[entry['dataset_idx']]}
                    else:
                        return entry
                dataset = dataset.map(initial_candidate_update)
            else:
                dataset = dataset.map(lambda entry: {**entry, 'candidates': entry['candidates'] + candidate_assign_dict[entry['dataset_idx']] if entry['dataset_idx'] in candidate_assign_dict else entry['candidates']})
            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Candidate Pool Update---Candidate Refinement")
            dataset, refinement_dataset, idx_refinement_mapping = self.refine_candidates(dataset=dataset, 
                                                                                            iter_num=iter_num,
                                                                                            beam_size=beam_size,
                                                                                            show_progress=show_progress,
                                                                                            modality_to_drop=modality_to_drop,
                                                                                            num_proc=num_proc)
            if return_initial_refinement:
                return dataset, refinement_dataset, idx_refinement_mapping
        else:
            dataset = precomputed_items['dataset']
            refinement_dataset = precomputed_items['refinement_dataset']
            idx_refinement_mapping = precomputed_items['idx_refinement_mapping']

        logging.info(f"Stage 3: Refinement Iteration {iter_num}---Candidate Pool Update---Calculate Candidate Metrics")
        dataset = dataset.map(lambda entry: calculate_metric_entry(entry), num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: normalize_rmse_metric(entry), num_proc=os.cpu_count() if num_proc is None else num_proc)
        logging.info(f"Stage 3: Refinement Iteration {iter_num}---Candidate Pool Update---Rerank and Filter Candidates")
        dataset = dataset.map(lambda entry: sort_candidates(entry, rerank_nmr_metric=rerank_nmr_metric, rerank_metric_ratio=rerank_metric_ratio, top_k=top_k, modality_to_drop=modality_to_drop), num_proc=os.cpu_count() if num_proc is None else num_proc)
        print('Modality to drop:', modality_to_drop)
        dataset = dataset.map(lambda entry: candidates_info_record(entry, iter_num=iter_num), num_proc=os.cpu_count() if num_proc is None else num_proc)
        
        # Scale down refinement dataset and mapping to only include selected candidates for next iteration
        if refinement_dataset and idx_refinement_mapping:
            update_refinement_entry_list = []
            update_idx_refinement_mapping = {}
            for i in range(len(dataset)):
                if dataset[i]['dataset_idx'] not in idx_refinement_mapping.keys():
                    continue
                refined_entries = idx_refinement_mapping[dataset[i]['dataset_idx']]
                refined_candidate_smiles = [entry['candidate'] for entry in refined_entries]
                new_candidates = [c for c in dataset[i]['candidates'] if c['type'] == 'refinement_' + str(iter_num)]
                if new_candidates:
                    update_idx_refinement_mapping[dataset[i]['dataset_idx']] = []
                    for c in new_candidates:
                        selected_entry = refined_entries[refined_candidate_smiles.index(c['smiles'])]
                        update_refinement_entry_list.append(selected_entry)
                        update_idx_refinement_mapping[dataset[i]['dataset_idx']].append(selected_entry)

            return dataset, Dataset.from_list(update_refinement_entry_list), update_idx_refinement_mapping
        else:
            return dataset, None, None
    
    def candidate_pool_construction(self,
                                    dataset: Dataset,
                                    top_k: int = 10, 
                                    beam_size: int = 10,
                                    show_progress: bool = False, 
                                    rerank_nmr_metric: str = 'rmse',
                                    rerank_metric_ratio: set = (1.0, 0.0, 0.0),
                                    modality_to_drop: Optional[str] = None,
                                    num_proc: int = None):
        # Take SMILES predicted by the nmr2mol model as initial candidate molecules
        logging.info(f"Stage 2: Candidate Pool Construction---Prepare Initial Candidate Molecules")
        candidate_list = []
        
        def pred_smiles_canonicalize(entry):
            original_pred_smiles = entry['pred_smiles']
            pred_smiles = list(set(entry['pred_smiles']))
            new_pred_smiles = []
            for s in pred_smiles:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    if smiles != s:
                        new_pred_smiles.append(s)
                        new_pred_smiles.append(smiles)
                    else:
                        new_pred_smiles.append(s)
                except:
                    new_pred_smiles.append(s)
            return {**entry, 'original_pred_smiles': original_pred_smiles, 'pred_smiles': new_pred_smiles}
        dataset = dataset.map(pred_smiles_canonicalize, num_proc=os.cpu_count() if num_proc is None else num_proc)
        
        for entry in dataset:
            pred_smiles = list(set(entry['pred_smiles']))
            entry_candidate_list = [{'smiles': s, 'dataset_idx': entry['dataset_idx'], 'type': 'prediction', 'source': ['None'], 'num': 1} for s in pred_smiles]
            candidate_list += entry_candidate_list

        
        # Generate NMR predictions for initial candidate molecules
        logging.info(f"Stage 2: Candidate Pool Construction---Generate Candidate NMR Predictions")
        candidate_with_nmr = self.candidate_nmr_gen(candidate_list, show_progress=show_progress, num_proc=num_proc)
        
        # Build candidate assignment dictionary
        candidate_assign_dict = {}
        candidate_assign_all_dict = {}
        idx_entry_mapping = {entry['dataset_idx']: entry for entry in dataset}
        unmatched_count = 0
        for i in range(len(candidate_with_nmr)):
            idx = candidate_with_nmr[i]['dataset_idx']
            target_molecule_formula = idx_entry_mapping[idx]['molecular_formula']
            candidate_molecule_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(candidate_with_nmr[i]['smiles']))
            if idx not in candidate_assign_all_dict:
                candidate_assign_all_dict[idx] = []
            candidate_assign_all_dict[idx].append({k: v for k, v in candidate_with_nmr[i].items() if k in ['smiles', 'type', 'source', 'num', 'predict', 'atoms']})
            if target_molecule_formula != candidate_molecule_formula:
                unmatched_count += 1
                continue
            else:
                if idx not in candidate_assign_dict:
                    candidate_assign_dict[idx] = []
                candidate_assign_dict[idx].append({k: v for k, v in candidate_with_nmr[i].items() if k in ['smiles', 'type', 'source', 'num', 'predict', 'atoms']})
        logging.info(f"Stage 2: Candidate Pool Construction---Ratio of candidate molecules with matched molecular formula: {len(candidate_with_nmr) - unmatched_count}/{len(candidate_with_nmr)}")

        # There are some entries without valid candidates predicted by the nmr2mol model after conformation construction filtering
        for i in range(len(dataset)):
            if dataset[i]['dataset_idx'] not in candidate_assign_dict:
                candidate_assign_dict[dataset[i]['dataset_idx']] = None
                candidate_assign_all_dict[dataset[i]['dataset_idx']] = None
        
        # Add initial candidate molecules to dataset entries
        logging.info(f"Stage 2: Candidate Pool Construction---Add Initial Candidate Molecules")
        def filter_uncanonical_smiles(entry_list):
            new_entry_list = []
            if entry_list is None:
                return None
            for entry in entry_list:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(entry['smiles']))
                    if smiles == entry['smiles']:
                        new_entry_list.append(entry)
                except:
                    continue
            return new_entry_list
        
        dataset = dataset.map(lambda entry: {**entry,
                                             'candidates_nmr2mol': candidate_assign_all_dict[entry['dataset_idx']], 
                                             'candidates': filter_uncanonical_smiles(candidate_assign_dict[entry['dataset_idx']])}, num_proc=os.cpu_count() if num_proc is None else num_proc)
        
        # Refine initial candidate molecules using the molref model
        logging.info(f"Stage 2: Candidate Pool Construction---Candidate Refinement")
        dataset, refinement_dataset, idx_refinement_mapping =  self.refine_candidates(dataset=dataset, 
                                                                                        iter_num=0,
                                                                                        beam_size=beam_size,
                                                                                        show_progress=show_progress,
                                                                                        modality_to_drop=modality_to_drop,
                                                                                        num_proc=num_proc)
        
        logging.info(f"Stage 2: Candidate Pool Construction---Calculate Candidate Metrics")
        dataset = dataset.map(lambda entry: calculate_metric_entry(entry) if entry['candidates'] is not None else entry, num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: normalize_rmse_metric(entry) if entry['candidates'] is not None else entry, num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: calculate_metric_entry(entry) if entry['candidates_nmr2mol'] is not None else entry, num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: normalize_rmse_metric(entry) if entry['candidates_nmr2mol'] is not None else entry, num_proc=os.cpu_count() if num_proc is None else num_proc)
        logging.info(f"Stage 2: Candidate Pool Construction---Rerank Candidates")
        dataset = dataset.map(lambda entry: sort_candidates(entry, rerank_nmr_metric=rerank_nmr_metric, rerank_metric_ratio=rerank_metric_ratio, top_k=top_k, modality_to_drop=modality_to_drop) if entry['candidates'] is not None else entry, num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: {**entry, 'history_valid_mols': ['']}, num_proc=os.cpu_count() if num_proc is None else num_proc)
        dataset = dataset.map(lambda entry: candidates_info_record(entry, iter_num=0), num_proc=os.cpu_count() if num_proc is None else num_proc)
        return dataset, refinement_dataset, idx_refinement_mapping
    
    def refine_candidates(self, 
                        dataset: Dataset, 
                        iter_num: int = 0,
                        beam_size: int = 10,
                        show_progress: bool = False,
                        modality_to_drop: Optional[str] = None,
                        num_proc: int = None):

        # Prepare dataset entries for refinement
        refinement_entry_list = []
        for i in range(len(dataset)):
            entry = dataset[i].copy()
            if iter_num == 0:
                candidates = list(set(dataset[i]['pred_smiles']))
                candidates_smiles = candidates
            else:
                candidates = dataset[i]['candidates']
                candidates_smiles = [c['smiles'] for c in candidates if c['type'] == 'refinement_' + str(iter_num)]
            for candidate_smiles in candidates_smiles:
                refinement_entry_list.append({'smiles': entry['smiles'], 
                                            'molecular_formula': entry['molecular_formula'],
                                            'h_nmr_peaks': entry['h_nmr_peaks'], 
                                            'c_nmr_peaks': entry['c_nmr_peaks'], 
                                            'candidate': candidate_smiles, 
                                            'dataset_idx': entry['dataset_idx']})
        if len(refinement_entry_list) == 0:
            return dataset, None, None
        
        # Refine candidate molecules using the molref model
        refinement_dataset = Dataset.from_list(refinement_entry_list)
        refined_smiles = self.molref_infer.infer_dataset(dataset=refinement_dataset, 
                                                         beam_size=beam_size, 
                                                         modality_to_drop=modality_to_drop,
                                                         show_progress=show_progress)

        refinement_dataset = refinement_dataset.map(lambda entry, i: {**entry, 'refined_smiles': refined_smiles[i]['pred'], 'refinement_score': refined_smiles[i]['scores']}, with_indices=True)
        
        # Build dataset entry to refinement entry mapping
        idx_refinement_mapping = {}
        for i in range(len(refinement_dataset)):
            idx = refinement_dataset[i]['dataset_idx']
            if idx not in idx_refinement_mapping.keys():
                idx_refinement_mapping[idx] = [refinement_dataset[i]]
            else:
                idx_refinement_mapping[idx].append(refinement_dataset[i])
        
        # Update dataset entries with refinement results and calculate OCS score(molref_score) for each candidate
        def candidates_refinement_update(entry, iter_num):
            if iter_num == 0:
                if entry['candidates_nmr2mol'] and entry['dataset_idx'] in idx_refinement_mapping.keys():
                    current_candidate_smiles = [c['smiles'] for c in entry['candidates_nmr2mol']]
                    for refinement_entry in idx_refinement_mapping[entry['dataset_idx']]:
                        if refinement_entry['candidate'] in current_candidate_smiles:
                            idx = current_candidate_smiles.index(refinement_entry['candidate'])
                            entry['candidates_nmr2mol'][idx]['refined_smiles']=refinement_entry['refined_smiles']
                            entry['candidates_nmr2mol'][idx]['refinement_score']=refinement_entry['refinement_score']
                            if entry['candidates_nmr2mol'][idx]['smiles'] in refinement_entry['refined_smiles']:
                                entry['candidates_nmr2mol'][idx]['molref_score'] = refinement_entry['refinement_score'][refinement_entry['refined_smiles'].index(entry['candidates_nmr2mol'][idx]['smiles'])]
                            else:
                                entry['candidates_nmr2mol'][idx]['molref_score'] = min(refinement_entry['refinement_score'])
            if entry['candidates'] and entry['dataset_idx'] in idx_refinement_mapping.keys():
                current_candidate_smiles = [c['smiles'] for c in entry['candidates']]
                for refinement_entry in idx_refinement_mapping[entry['dataset_idx']]:
                    if refinement_entry['candidate'] in current_candidate_smiles:
                        idx = current_candidate_smiles.index(refinement_entry['candidate'])
                        entry['candidates'][idx]['refined_smiles']=refinement_entry['refined_smiles']
                        entry['candidates'][idx]['refinement_score']=refinement_entry['refinement_score']
                        if entry['candidates'][idx]['smiles'] in refinement_entry['refined_smiles']:
                            entry['candidates'][idx]['molref_score'] = refinement_entry['refinement_score'][refinement_entry['refined_smiles'].index(entry['candidates'][idx]['smiles'])]
                        else:
                            entry['candidates'][idx]['molref_score'] = min(refinement_entry['refinement_score'])
            return entry
        dataset = dataset.map(candidates_refinement_update, fn_kwargs={'iter_num': iter_num}, num_proc=os.cpu_count() if num_proc is None else num_proc)
        # dataset = dataset.map(lambda entry: candidates_refinement_update(entry, iter_num=iter_num), num_proc=os.cpu_count() if num_proc is None else num_proc)
        # data_list = []
        # for i in range(len(dataset)):
        #     entry = candidates_refinement_update(dataset[i], iter_num=iter_num)
        #     data_list.append(entry)
        # dataset = Dataset.from_list(data_list)
        return dataset, refinement_dataset, idx_refinement_mapping
    
    def infer_dataset(self, 
                      dataset: Dataset, 
                      show_progress: bool = False, 
                      enable_dataset_progress: bool = False,
                      top_k: int = 10,
                      beam_size: int = 10, 
                      refinement_iters: int = 1,
                      rerank_nmr_metric: str = 'rmse',
                      rerank_metric_ratio: tuple = (1.0, 0.0, 0.0),
                      modality_to_drop: str = None,
                      num_proc: int = None,
                      return_initial_refinement: bool = False):
        # Check input parameters
        if modality_to_drop not in [None, 'Multiplets', 'Carbon']:
            raise ValueError("modality_to_drop must be one of None, 'Multiplets', or 'Carbon'.")
        if modality_to_drop is not None:
            assert len(rerank_metric_ratio) == 2, "rerank_metric_ratio should be a tuple of (nmr_weight, molref_score_weight)."
        assert top_k >= beam_size, "top_k should be larger than or equal to beam_size"
        if not enable_dataset_progress:
            disable_progress_bar()
        else:
            enable_progress_bar()
        
        # Add dataset indices if not present
        if 'dataset_idx' not in dataset.features:
            dataset = dataset.map(lambda entry, idx: {**entry, 'dataset_idx': idx}, with_indices=True)
        
        # Stage 1: generate initial candidate molecules from NMR spectra with nmr2mol model
        logging.info(f"Stage 1: Initial Candidate Molecule Generation")
        pred_smiles = self.nmr2mol_infer.infer_dataset(dataset = dataset, beam_size=beam_size, modality_to_drop=modality_to_drop, show_progress=show_progress)
        dataset = dataset.map(lambda entry, idx: {**entry, 'pred_smiles': pred_smiles[idx]['pred'], 'pred_scores': pred_smiles[idx]['scores']}, with_indices=True)

        # Stage 2: construct candidate pool with mol2nmr model
        logging.info(f"Stage 2: Candidate Pool Construction")
        dataset, refinement_dataset, idx_refinement_mapping = self.candidate_pool_construction(dataset, 
                                                                                                top_k=top_k, 
                                                                                                beam_size=beam_size,
                                                                                                show_progress=show_progress, 
                                                                                                rerank_nmr_metric=rerank_nmr_metric,
                                                                                                rerank_metric_ratio=rerank_metric_ratio,
                                                                                                modality_to_drop=modality_to_drop,
                                                                                                num_proc=num_proc)

        logging.info(f"Stage 3: Iterative Candidate Pool Refinement")
        for iter_num in range(1, refinement_iters+1):
            if refinement_dataset is None:
                break
            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Curate New Candidates List")
            
            # Build dataset index to entries mapping and dataset index to entry index mapping
            idx_entry_mapping = {entry['dataset_idx']: entry for entry in dataset}
            idx_i_mapping = {dataset[i]['dataset_idx']: i for i in range(len(dataset))}

            # Curate new candidate molecules from refinement results
            candidate_list = []
            candidate_smiles_list = []
            candidate_source_update_list = [[[]]*len(entry['candidates']) if entry['candidates'] is not None else [] for entry in dataset]
            
            for idx, refinement_entries in idx_refinement_mapping.items():
                if idx_entry_mapping[idx]['candidates'] is None:
                    continue
                prev_candidate_smiles = [c['smiles'] for c in idx_entry_mapping[idx]['candidates']]
                history_valid_mols = idx_entry_mapping[idx]['history_valid_mols'] + prev_candidate_smiles
                current_candidate_smiles = []
                
                for j in range(len(refinement_entries)):
                    for k in range(len(refinement_entries[j]['refined_smiles'])):
                        try:
                            refined_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(refinement_entries[j]['refined_smiles'][k]))
                        except:
                            continue
                        if refined_smiles not in history_valid_mols and refined_smiles not in current_candidate_smiles:
                            current_candidate_smiles.append(refined_smiles)
                            candidate_list.append({'smiles': refined_smiles, 'dataset_idx': idx, 'type': 'refinement_' + str(iter_num), 'source': [refinement_entries[j]['candidate']+"_"+str(iter_num-1)]})
                            candidate_smiles_list.append(refined_smiles)
                        else:
                            if refined_smiles in prev_candidate_smiles:
                                candidate_index = prev_candidate_smiles.index(refined_smiles)
                                candidate_source_update_list[idx_i_mapping[idx]][candidate_index].append(refinement_entries[j]['candidate']+"_"+str(iter_num-1)) if (refinement_entries[j]['candidate']+"_"+str(iter_num-1)) not in candidate_source_update_list[idx_i_mapping[idx]][candidate_index] else None

                            if refined_smiles in current_candidate_smiles:
                                candidate_index = candidate_smiles_list.index(refined_smiles)
                                candidate_list[candidate_index]['source'].append(refinement_entries[j]['candidate']+"_"+str(iter_num-1)) if (refinement_entries[j]['candidate']+"_"+str(iter_num-1)) not in candidate_list[candidate_index]['source'] else None

            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Update Source of Previous Candidates")
            def update_candidate_num(entry):
                if entry['candidates'] is None:
                    return entry
                for i in range(len(entry['candidates'])):
                    entry['candidates'][i]['source'] += candidate_source_update_list[idx_i_mapping[entry['dataset_idx']]][i]
                return entry
            dataset = dataset.map(update_candidate_num, num_proc=os.cpu_count() if num_proc is None else num_proc)

            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Candidate Pool Update")
            dataset, refinement_dataset, idx_refinement_mapping = self.candidate_pool_update(dataset=dataset, 
                                                                                            candidate_list=candidate_list, 
                                                                                            top_k=top_k, 
                                                                                            beam_size=beam_size,
                                                                                            show_progress=show_progress, 
                                                                                            rerank_nmr_metric=rerank_nmr_metric, 
                                                                                            rerank_metric_ratio=rerank_metric_ratio, 
                                                                                            iter_num=iter_num,
                                                                                            modality_to_drop=modality_to_drop,
                                                                                            return_initial_refinement=return_initial_refinement,
                                                                                            num_proc=os.cpu_count() if num_proc is None else num_proc)

            if return_initial_refinement:
                return dataset, refinement_dataset
        return dataset
    
    def refine_dataset(self, 
                      dataset: Dataset, 
                      refinement_dataset: Dataset = None,
                      idx_refinement_mapping: dict = None,
                      show_progress: bool = False, 
                      enable_dataset_progress: bool = False,
                      top_k: int = 10,
                      beam_size: int = 10, 
                      refinement_iters: int = 5,
                      rerank_nmr_metric: str = 'rmse',
                      rerank_metric_ratio: set = (1.0, 0.0, 0.0),
                      modality_to_drop: str = None,
                      num_proc: int = None):
        for iter_num in range(2, refinement_iters+1):
            if refinement_dataset is None:
                break
            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Curate New Candidates List")
            
            # Build dataset index to entries mapping and dataset index to entry index mapping
            idx_entry_mapping = {entry['dataset_idx']: entry for entry in dataset}
            idx_i_mapping = {dataset[i]['dataset_idx']: i for i in range(len(dataset))}

            # Curate new candidate molecules from refinement results
            candidate_list = []
            candidate_smiles_list = []
            candidate_source_update_list = [[[]]*len(entry['candidates']) if entry['candidates'] is not None else [] for entry in dataset]
            
            for idx, refinement_entries in idx_refinement_mapping.items():
                if idx_entry_mapping[idx]['candidates'] is None:
                    continue
                prev_candidate_smiles = [c['smiles'] for c in idx_entry_mapping[idx]['candidates']]
                history_valid_mols = idx_entry_mapping[idx]['history_valid_mols'] + prev_candidate_smiles
                current_candidate_smiles = []
                
                for j in range(len(refinement_entries)):
                    for k in range(len(refinement_entries[j]['refined_smiles'])):
                        try:
                            refined_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(refinement_entries[j]['refined_smiles'][k]))
                        except:
                            continue
                        if refined_smiles not in history_valid_mols and refined_smiles not in current_candidate_smiles:
                            current_candidate_smiles.append(refined_smiles)
                            candidate_list.append({'smiles': refined_smiles, 'dataset_idx': idx, 'type': 'refinement_' + str(iter_num), 'source': [refinement_entries[j]['candidate']+"_"+str(iter_num-1)]})
                            candidate_smiles_list.append(refined_smiles)
                        else:
                            if refined_smiles in prev_candidate_smiles:
                                candidate_index = prev_candidate_smiles.index(refined_smiles)
                                candidate_source_update_list[idx_i_mapping[idx]][candidate_index].append(refinement_entries[j]['candidate']+"_"+str(iter_num-1)) if (refinement_entries[j]['candidate']+"_"+str(iter_num-1)) not in candidate_source_update_list[idx_i_mapping[idx]][candidate_index] else None

                            if refined_smiles in current_candidate_smiles:
                                candidate_index = candidate_smiles_list.index(refined_smiles)
                                candidate_list[candidate_index]['source'].append(refinement_entries[j]['candidate']+"_"+str(iter_num-1)) if (refinement_entries[j]['candidate']+"_"+str(iter_num-1)) not in candidate_list[candidate_index]['source'] else None

            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Update Source of Previous Candidates")
            def update_candidate_num(entry):
                if entry['candidates'] is None:
                    return entry
                for i in range(len(entry['candidates'])):
                    entry['candidates'][i]['source'] += candidate_source_update_list[idx_i_mapping[entry['dataset_idx']]][i]
                return entry
            dataset = dataset.map(update_candidate_num, num_proc=os.cpu_count() if num_proc is None else num_proc)

            logging.info(f"Stage 3: Refinement Iteration {iter_num}---Candidate Pool Update")
            dataset, refinement_dataset, idx_refinement_mapping = self.candidate_pool_update(dataset=dataset, 
                                                                                            candidate_list=candidate_list, 
                                                                                            top_k=top_k, 
                                                                                            beam_size=beam_size,
                                                                                            show_progress=show_progress, 
                                                                                            rerank_nmr_metric=rerank_nmr_metric, 
                                                                                            rerank_metric_ratio=rerank_metric_ratio, 
                                                                                            iter_num=iter_num,
                                                                                            modality_to_drop=modality_to_drop,
                                                                                            num_proc=os.cpu_count() if num_proc is None else num_proc)


        return dataset
