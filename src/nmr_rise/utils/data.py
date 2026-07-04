from datasets import load_dataset, DatasetDict, load_from_disk
import os
import json
from ast import literal_eval
import random
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from rxn.chemutils.tokenization import tokenize_smiles
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional

def candidate_aug(batch, aug_num=5):
    """
    Batch version of candidate augmentation function.
    For each example in the batch, samples aug_num candidates from the 'candidates' list,
    then expands each example into multiple rows - one per sampled candidate.
    
    Args:
        batch (dict): A batch of examples as lists for each column.
        aug_num (int): Number of candidates to sample and expand per example.
    
    Returns:
        dict: Augmented batch with each example expanded into multiple rows.
    """
    aug_rows = {
        k: [] for k in batch if k != 'candidates'
    }
    aug_rows['candidate'] = []
    aug_rows['aug_id'] = []
    
    for i, example_candidates_str in enumerate(batch['candidates']):
        candidates = literal_eval(example_candidates_str)
        # Sample candidates with or without replacement
        if len(candidates) < aug_num:
            sampled_candidates = random.choices(candidates, k=aug_num)
        else:
            sampled_candidates = random.sample(candidates, aug_num)
        
        for j, candidate in enumerate(sampled_candidates):
            # Copy all columns except 'candidates'
            for key in batch:
                if key != 'candidates':
                    aug_rows[key].append(batch[key][i])
            aug_rows['candidate'].append(candidate)
            aug_rows['aug_id'].append(f"aug_{batch.get('idx', [''])[i]}_{j}")

    return aug_rows

def dataset_aug_molref(dataset, aug_num=5):
    """
    Augments the dataset for training the MolRef model by expanding each example into multiple rows based on candidates.
    
    Args:
        dataset (datasets.Dataset): HuggingFace dataset to augment.
        aug_num (int): Number of augmented samples per example.
    
    Returns:
        datasets.Dataset: Augmented dataset with expanded rows and extra columns.
    """
    return dataset.map(
        lambda batch: candidate_aug(batch, aug_num=aug_num),
        batched=True,
        batch_size=1,  # Process one example at a time for expansion
        remove_columns=['candidates'],
        num_proc=os.cpu_count()
    )

def data_process(data_path, aug_num=5, shuffle_seed=42):
    def load_split(data_dir, split):
        ds = load_dataset('parquet', data_dir=os.path.join(data_dir, split))
        return next(iter(ds.values()))

    data_dir = os.path.join(data_path, 'split')
    splits = ['train', 'valid', 'test']
    dataset = DatasetDict({k if k != 'valid' else 'validation': load_split(data_dir, k) for k in splits})

    dataset.save_to_disk(os.path.join(data_path, 'full'))

    def load_results(filename):
        with open(os.path.join(data_path, filename), 'r') as f:
            return {item['true']: str(item['pred']) for item in json.load(f)}

    lookups = {
        'train': load_results('multitask_C_H_train_results.json'),
        'validation': load_results('multitask_C_H_valid_results.json'),
        'test': load_results('multitask_C_H_test_results.json')
    }

    def add_candidates(example, lookup):
        example['candidates'] = lookup.get(example['smiles'], "")
        return example

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda ex: add_candidates(ex, lookups[split]))

    train_aug = dataset_aug_molref(dataset['train'], aug_num=aug_num)
    val_aug = dataset_aug_molref(dataset['validation'], aug_num=aug_num)
    test_aug = dataset_aug_molref(dataset['test'], aug_num=aug_num)
    # shuffle the augmented datasets
    train_aug = train_aug.shuffle(seed=shuffle_seed)
    val_aug = val_aug.shuffle(seed=shuffle_seed)
    test_aug = test_aug.shuffle(seed=shuffle_seed)
    aug_dataset = DatasetDict({
        'train': train_aug,
        'validation': val_aug,
        'test': test_aug
    })
    aug_dataset.save_to_disk(os.path.join(data_path, f'revision_{aug_num}'))
    return dataset, aug_dataset

def conformation_construction(
        entry: dict,
        seed: int = 42,
    ) -> dict:
    """
    Conformation Construction
    Convert a SMILES into a 3-D RDKit record with coordinates
    and placeholder target arrays.
    """
    try:
        # 1. Parse SMILES and add hydrogens
        mol = Chem.MolFromSmiles(entry['smiles'])
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        # 2. Embed (use ETKDGv3 if no custom params supplied)
        status = AllChem.EmbedMolecule(mol, params = params)
        if status == -1:
            raise ValueError("Comformer generation failed")
        # 3. MMFF94 geometry optimisation to refine coordinates
        try:
            # some conformer can not use MMFF optimize
            AllChem.MMFFOptimizeMolecule(mol)
            coords = mol.GetConformer().GetPositions().astype(np.float32)
        except:
            coords = mol.GetConformer().GetPositions().astype(np.float32)
        
        # 4. Extract atomic symbols
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # 5. Build placeholder property arrays (edit as needed)
        atoms_target = np.zeros(len(atoms), dtype=float)       
        # label 1 to C, H atoms, 0 to others
        atoms_target_mask = np.zeros(len(atoms), dtype=int)    # 0 =
        for i, atom in enumerate(atoms):
            if atom in ["C", "H"]:
                atoms_target_mask[i] = 1

        # 6. Compute InChIKey (needs rdkit-inchi support)
        inchikey = inchi.MolToInchiKey(mol)
        # 7. Update entry with new fields
        entry.update({
            "atoms": atoms,
            "coordinates": coords.tolist(),
            "atoms_target": atoms_target.tolist(),
            "atoms_target_mask": atoms_target_mask.tolist(),
            "smiles_with_hydrogens": Chem.MolToSmiles(mol, isomericSmiles=True),
            "inchikey": inchikey,
            "is_converted": True
        })
    except Exception as e:
        entry.update({
            "atoms": None,
            "coordinates": None,
            "atoms_target": None,
            "atoms_target_mask": None,
            "smiles_with_hydrogens": None,
            "inchikey": None,
            "is_converted": False
        })

    return entry

def filter_invalid_smiles(entry: dict) -> bool:
    try:
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol is None:
            return False
        return True
    except:
        return False

def filter_invalid_entry(entry: dict) -> bool:
    flag = True
    if not entry["is_converted"]:
        flag = False
    # if not sum(entry["atoms_target_mask"]):
    #     flag = False
    return flag

def match_and_fill(exp_peak_list: List[float], pred_peak_list: List[float]) -> List[float]:
    exp_len, pred_len = len(exp_peak_list), len(pred_peak_list)
    # 记录所有候选(距离)
    candidates = []
    for i, e in enumerate(exp_peak_list):
        for j, p in enumerate(pred_peak_list):
            dist = abs(e - p)
            candidates.append((i, j, dist))
    # 按距离排序
    candidates.sort(key=lambda x: x[2])


    # 贪心分配
    matched_exp = set()
    matched_pred = set()
    match_res = [None] * pred_len
    for i, j, dist in candidates:
        if i not in matched_exp and j not in matched_pred:
            match_res[j] = exp_peak_list[i]
            matched_exp.add(i)
            matched_pred.add(j)
        if len(matched_exp) == exp_len:
            break

    if len(exp_peak_list) < len(pred_peak_list):
        # 未分配的pred元素补齐
        fill = [pred_peak_list[j] for j in range(pred_len) if j not in matched_pred]
        padded_result = exp_peak_list + fill
        return sorted(padded_result, reverse=True)
    else:
        return sorted(match_res, reverse=True)

def peak_match(exp_peak_list, pred_peak_list):
    """
    使用最小费用匹配算法（匈牙利算法）来找到总距离最小的匹配结果
    
    相比原始贪心算法的改进：
    1. 保证全局最优解（总距离最小）
    2. 使用scipy的高效实现
    3. 时间复杂度O(n³)
    
    参数:
    exp_peak_list: 实验峰值列表
    pred_peak_list: 预测峰值列表
    
    返回:
    match_res: 匹配结果字典，格式与原函数相同
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    exp_len, pred_len = len(exp_peak_list), len(pred_peak_list)
    
    # 创建费用矩阵 - 每个元素是对应实验峰值与预测峰值的距离
    cost_matrix = np.zeros((exp_len, pred_len))
    for i, exp_peak in enumerate(exp_peak_list):
        for j, pred_peak in enumerate(pred_peak_list):
            cost_matrix[i, j] = abs(exp_peak - pred_peak)
    
    # 使用scipy的线性分配算法（匈牙利算法的优化实现）求最优解
    # 返回行索引和列索引，使得总费用最小
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 构建结果字典，保持与原始函数相同的输出格式
    match_res = {}
    for k, (i, j) in enumerate(zip(row_indices, col_indices)):
        match_res[k] = {
            'exp_peak': exp_peak_list[i], 
            'pred_peak': pred_peak_list[j], 
            'dist': cost_matrix[i, j]
        }
    return match_res

def hnmr_list_generation(hnmr_list: List[Dict]) -> List[float]:
    new_hnmr_list = []
    for i in range(len(hnmr_list)):
        for j in range(hnmr_list[i]['nH']):
            # randomly sample a peak in the range of min and max
            if hnmr_list[i]['delta']:
                new_hnmr_list.append(hnmr_list[i]['delta'])
            else:
                new_hnmr_list.append(random.uniform(hnmr_list[i]['rangeMin'], hnmr_list[i]['rangeMax']))
    return new_hnmr_list

def mol2nmr_target_generation(
    sample: dict,
) -> List[dict]:
    """
    将实验峰值与预测峰值进行匹配，并生成最终的目标列表
    """
    exp_C = [sample['c_nmr_peaks'][i]['delta (ppm)'] for i in range(len(sample['c_nmr_peaks']))]
    exp_H = hnmr_list_generation(sample['h_nmr_peaks'])
    pred_peak_list = sample['predict']
    atoms = sample['atoms']
    # 分离C和H的预测峰值
    pred_C = [p for p, a in zip(pred_peak_list, atoms) if a == "C"]
    pred_H = [p for p, a in zip(pred_peak_list, atoms) if a == "H"]

    # 分别进行峰值匹配
    matched_C = peak_match(exp_C, pred_C)
    matched_H = peak_match(exp_H, pred_H)

    # 合并匹配结果
    atoms_target = pred_peak_list.copy()
    # 更新pred_peak_list中的匹配信息
    for _, peak in matched_C.items():
        idx = atoms_target.index(peak['pred_peak'])
        atoms_target[idx] = peak['exp_peak']
    for _, peak in matched_H.items():
        idx = atoms_target.index(peak['pred_peak'])
        atoms_target[idx] = peak['exp_peak']
    sample['atoms_target'] = atoms_target
    return sample

def calculate_rmse(sample):
    """Calculate RMSE between experimental and predicted chemical shifts"""
    
    # Calculate RMSE
    target = sample['atoms_target']
    pred = sample['predict']
    atoms = sample['atoms']
    C_peaks_tgt = [t for t, a in zip(target, atoms) if a == "C"]
    H_peaks_tgt = [t for t, a in zip(target, atoms) if a == "H"]
    C_peaks_pred = [p for p, a in zip(pred, atoms) if a == "C"]
    H_peaks_pred = [p for p, a in zip(pred, atoms) if a == "H"]
    C_squared_diff = [(t - p)**2 for t, p in zip(C_peaks_tgt, C_peaks_pred)]
    H_squared_diff = [(t - p)**2 for t, p in zip(H_peaks_tgt, H_peaks_pred)]


    # squared_diff = [(v['exp_peak'] - v['pred_peak'])**2 for v in matched_data.values]
    C_rmse = np.sqrt(np.mean(C_squared_diff)) if len(C_squared_diff) > 0 else 0.0
    H_rmse = np.sqrt(np.mean(H_squared_diff)) if len(H_squared_diff) > 0 else 0.0
    # return rmse with 2 decimal places
    sample['C_rmse'] = round(C_rmse, 2)
    sample['H_rmse'] = round(H_rmse, 2)
    return sample

def set_match_score_weighted(x: np.ndarray, y: np.ndarray, weight_matrix: Optional[np.ndarray] = None, sigma: float = 1) -> float:
    """
    Computes a weighted matching score between two sets of NMR data.
    """
    if len(x) == 0 or len(y) == 0:
        return 1.0 if (len(x) == 0 and len(y) == 0) else 0.0
    
    dist_matrix = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
    score_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    if weight_matrix is not None:
        assert score_matrix.shape == weight_matrix.shape
        score_matrix *= weight_matrix
    
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = score_matrix[row_ind, col_ind].sum()
    
    score = total_score / np.sqrt(len(x) * len(y))
    return score

def set2vec(set_list: List, nmr_type: str, dim: int = 128, normalize: bool = True, sigma: Optional[float] = None) -> np.ndarray:
    """
    Converts a set of NMR data into a vector representation.
    """
    if isinstance(set_list, np.ndarray):
        return set2vec([set_list], nmr_type, dim, normalize)
    # set_list: list of np.array
    assert isinstance(set_list, list)
    if nmr_type == 'H':
        nmr_range = (-1, 15)
        sigma = sigma or 0.3
    elif nmr_type == 'C':
        nmr_range = (-10, 230)
        sigma = sigma or 2
    ni = np.linspace(nmr_range[0], nmr_range[1], dim)
    interval = ni[1] - ni[0]
    # gaussian kernel
    coef = interval / (np.sqrt(2 * np.pi) * sigma)
    
    result = [coef * np.exp(-(np.abs(item[:, np.newaxis] - ni) / sigma) ** 2 / 2).sum(axis=0) for item in set_list]
    
    result = np.array(result)
    if normalize:
        return normalize_vectors(result).astype(np.float32)
    else:
        return result.astype(np.float32)

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Euclidean distance between two vectors.
    """
    return np.linalg.norm(a - b)