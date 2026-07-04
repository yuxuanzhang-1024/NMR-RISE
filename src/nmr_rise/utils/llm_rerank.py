import multiprocessing as mp
import time
import random
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import queue
import threading
from typing import List, Dict, Any
import os
from datasets import Dataset
import logging
import traceback
from typing import List, Dict, Union, Optional
import re
from tqdm import tqdm
from perplexity import Perplexity
import base64
import copy
from matplotlib import pyplot as plt
from PIL import Image
import json
from rdkit import Chem
from rdkit.Chem import Draw

def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:
    multiplet_str = "1HNMR "
    for peak in multiplets:
        range_max = float(peak["rangeMax"]) 
        range_min = float(peak["rangeMin"]) 

        formatted_peak = ""
        formatted_peak = formatted_peak + "{:.2f} {:.2f} ".format(range_max, range_min)        
        formatted_peak = formatted_peak +  "{} {}H ".format(
                                                            peak["category"],
                                                            peak["nH"],
                                                        )
        js = str(peak["j_values"])
        if js != "None":
            split_js = js.split("_")
            split_js = list(filter(None, split_js))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_js = "J " + " ".join(processed_js)
            formatted_peak += formatted_js

        multiplet_str += formatted_peak.strip() + " | "

    # Remove last separating token
    multiplet_str = multiplet_str[:-2]
    return multiplet_str

def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    nmr_string = "13CNMR "
    for peak in carbon_nmr:
        nmr_string += str(round(float(peak["delta (ppm)"]), 1)) + " "

    return nmr_string

def tokenize_formula(formula: str) -> list:
    return ' '.join(re.findall("[A-Z][a-z]?|\d+|.", formula)) + ' '

def nmr_data_gen(entry):
    hnmr_str = process_hnmr(entry['h_nmr_peaks'])
    cnmr_str = process_cnmr(entry['c_nmr_peaks'])
    formula_str = tokenize_formula(entry['molecular_formula'])
    return (formula_str + hnmr_str + cnmr_str).strip()

def candidate_gen(entry):
    mol = Chem.MolFromSmiles(entry['smiles'])
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()         # 0-based
        atom.SetAtomMapNum(idx + 1) # 比如设成 1..N

    mapped_smi = Chem.MolToSmiles(mol)
    pattern = r'(\[[^\[\]:]+:(\d+)\])'
    matches = re.findall(pattern, mapped_smi)
    matches = {match[0]: int(match[1]) for match in matches}
    predict_nmr = entry['predict']
    nmr_mapping = {key: round(predict_nmr[val-1], 2) for key, val in matches.items()}
    return mapped_smi, nmr_mapping

def draw_mol(smiles, idx, save_dir_path=None):
    plt.figure(figsize=(4, 4), dpi=500)
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(800, 800))  # Larger size = higher resolution
    plt.text(0.0, 0.9, f'({idx+1})', transform=plt.gca().transAxes, fontsize=9)
    plt.imshow(img)
    plt.axis('off')
    save_path = f'{save_dir_path}/{idx+1}.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return save_path

def llm_rerank_data_gen(entry, prompt_template_path, save_path):
    save_dir_path = f'{save_path}/{entry["dataset_idx"]}_{entry["idx"]}'
    os.makedirs(save_dir_path, exist_ok=True)
    img_save_dir_path = f'{save_dir_path}/images'
    os.makedirs(img_save_dir_path, exist_ok=True)
    candidates = entry['candidates'].copy()
    candidates_smiles = [c['smiles'] for c in candidates]
    nmr_data = nmr_data_gen(entry)
    mapped_smi_list = []
    candidate_info_list = []
    img_path_list = []
    for i, candidate in enumerate(candidates):
        mapped_smi, nmr_mapping = candidate_gen(candidate)
        mapped_smi_list.append(mapped_smi)
        ces = 0.6*candidate['cnmr_rmse_normalized'] + 0.3*candidate['hnmr_rmse_normalized'] + 0.1*candidate['molref_score']
        candidate_info_list.append({
            'SMILES': candidate['smiles'],
            'Predicted Atom-Mapping NMR Peaks': nmr_mapping,
            "CNMR Normalized RMSE Score": round(candidate['cnmr_rmse_normalized'], 4),
            "HNMR Normalized RMSE Score": round(candidate['hnmr_rmse_normalized'], 4), 
            "Optimization Completeness Score": round(candidate['molref_score'], 4), 
            "Comprehensive Evaluation Score": round(ces, 4)
        })
        img_path_list.append(draw_mol(mapped_smi, i, img_save_dir_path))
    img_list = [Image.open(img_path).convert("RGB") for img_path in img_path_list]
    img_list[0].save(f'{save_dir_path}/candidate_structure_images.pdf', save_all=True, append_images=img_list[1:])
    
    candidates_info_summary = '\n'.join([json.dumps(item, ensure_ascii=False) for item in candidate_info_list])
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    prompt = prompt_template.format(nmr_data=nmr_data, candidates_info_summary=candidates_info_summary)
    target_smiles = entry['smiles']
    with open(f'{save_dir_path}/prompt.json', 'w') as f:
        json.dump({'idx': entry['idx'], 'smiles': target_smiles, 'candidates_smiles': candidates_smiles, 'prompt': prompt, 'output': ''}, f, ensure_ascii=False, indent=4)
    return entry, img_list, prompt

class PerplexityReranker:
    """
    A thread-safe LLM reranker for molecular candidates using token rotation and multiprocessing.
    """
    
    def __init__(self, 
                 api_tokens: List[str], 
                 max_workers: Optional[int] = None,
                 history_conversation_pool: List[Dict[str, Any]] = None,
                 timeout: int = 300):
        """
        Initialize the LLM reranker.
        
        Args:
            api_tokens: List of API tokens for load balancing
            max_workers: Maximum number of concurrent workers (defaults to number of tokens)
            log_path: Directory to store logs
            timeout: API request timeout in seconds
        """
        if not api_tokens:
            raise ValueError("At least one API token must be provided")
            
        self.api_tokens = api_tokens
        self.max_workers = max_workers or min(len(api_tokens), 8)  # Cap at 8 workers
        self.timeout = timeout
        
        # Thread-safe token queue for load balancing
        self.available_tokens = queue.Queue()
        self.token_lock = threading.Lock()
        
        # Supported model types
        self.model_type_list = [
            "sonar",
            "sonar-pro", 
            "sonar-reasoning",
            "sonar-reasoning-pro",
        ]
        
        # Initialize token queue
        self._initialize_token_queue()
        self.history_conversation_pool = history_conversation_pool or []
   
    def _initialize_token_queue(self):
        """Initialize the token queue with available tokens."""
        for token in self.api_tokens:
            self.available_tokens.put(token)
            
    def _get_token(self) -> str:
        """
        Get an available token from the queue in a thread-safe manner.
        
        Returns:
            Available API token
        """
        try:
            # Use timeout to avoid indefinite blocking
            token = self.available_tokens.get(timeout=30)
            return token
        except queue.Empty:
            raise RuntimeError("No API tokens available - all tokens may be in use")
            
    def _return_token(self, token: str):
        """
        Return a token to the available queue.
        
        Args:
            token: API token to return
        """
        with self.token_lock:
            self.available_tokens.put(token)
            
    def _validate_llm_type(self, llm_type: str):
        """
        Validate the LLM model type.
        
        Args:
            llm_type: The model type to validate
            
        Raises:
            ValueError: If llm_type is not supported
        """
        if llm_type not in self.model_type_list:
            raise ValueError(
                f"Invalid llm_type: {llm_type}. "
                f"Supported types are: {self.model_type_list}"
            )
    
    def result_text_check(self, result_text: str, entry: dict, check_target: bool) -> bool:
        """
        Check if the result_text contains the expected reranked key.
        
        Args:
            result_text: The text output from the LLM
        """
        result_dict = {
                'result_text': result_text,
                'reranked_dict': {},
                'reranked_smiles': []
            }
        try:
            candidates_smiles = entry['candidates_smiles']
            # reasoning_text = result_text.split('</think>')[0] + '</think>'
            # reranked_text = result_text.split('</think>')[1]
            reranked_text = result_text
            if "```python" in reranked_text:
                reranked_result = reranked_text.split("```python\n")[1].split("```")[0]
            else:
                reranked_result = reranked_text.split("```json\n")[1].split("```")[0]
            reranked_dict = json.loads(reranked_result)
            key_to_rename = None
            for key in reranked_dict.keys():
                if 'Reranked' in key and key != 'Reranked Molecular Smiles':
                    key_to_rename = key
                    break
            if key_to_rename:
                reranked_dict['Reranked Molecular Smiles'] = reranked_dict[key_to_rename]
                del reranked_dict[key_to_rename]
            if 'Reranked Molecular Smiles' not in reranked_dict:
                print(f'No reranked key found: idx {entry["idx"]}')
                return False, result_dict
            else:
                reranked_smiles = reranked_dict['Reranked Molecular Smiles']
                if set(reranked_smiles) != set(candidates_smiles):
                    print(f"Reranked SMILES do not match candidates: idx {entry['idx']}")
                    return False, result_dict
                if check_target:
                    target_smiles = entry['smiles']
                    if target_smiles in candidates_smiles and target_smiles in reranked_smiles:
                        if candidates_smiles.index(target_smiles) < reranked_smiles.index(target_smiles):
                            print(f"Target rank not improved: idx {entry['idx']}")
                            return False, result_dict
                # result_dict['reasoning_text'] = reasoning_text
                result_dict['reranked_dict'] = reranked_dict
                result_dict['reranked_smiles'] = reranked_smiles
                return True, result_dict
        except Exception as e:
            print(e)
            return False, result_dict
    
    def multi_conversation_curation(self, history_conversation_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = []
        for conv in history_conversation_list:
            mol_image_pdf_path = conv['mol_image_pdf_path']
            with open(mol_image_pdf_path, "rb") as file:
                file_data = file.read()
                pdf_file = base64.b64encode(file_data).decode('utf-8')
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": conv['prompt']},
                    {"type": "file_url",
                     "file_url": {"url": copy.deepcopy(pdf_file)},
                     "file_name": "candidate_structure_images.pdf"}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": conv['result_text']
            })
        return messages
    
    def llm_rerank(self, entry: Dict[str, Any], llm_type: str, trials: int, 
                check_target: bool, use_history_conversation: bool, 
                history_conversation_num: int) -> Dict[str, Any]:
        """Rerank candidate molecules using LLM with 5-minute timeout."""
        
        api_token = None
        
        try:
            self._validate_llm_type(llm_type)
            
            idx = entry['idx']
            prompt = entry['prompt']
            mol_image_pdf_path = entry['mol_image_pdf_path']
            
            api_token = self._get_token()
                
            if mol_image_pdf_path:
                with open(mol_image_pdf_path, "rb") as file:
                    file_data = file.read()
                    pdf_file = base64.b64encode(file_data).decode('utf-8')
            
            try:
                client = Perplexity(api_key=api_token, timeout=self.timeout)
            except ImportError:
                raise ImportError("Required Perplexity client library not found")
            
            while trials > 0:
                trials -= 1
                
                # 初始化 messages - 系统消息
                messages = [{
                    "role": "system", 
                    "content": "You are a chemist specialized in NMR spectrum analysis."
                }]
                
                # 添加历史对话示例
                if use_history_conversation:
                    history_conversation = random.sample(
                        self.history_conversation_pool, 
                        min(history_conversation_num, len(self.history_conversation_pool))
                    )
                    # ✓ 使用 extend() 而不是 append()
                    history_messages = self.multi_conversation_curation(history_conversation)
                    messages.extend(history_messages)  # ← 关键修改！

                # 添加当前用户查询
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "file_url",
                            "file_url": {
                                "url": pdf_file,
                            },
                            "file_name": "candidate_structure_images.pdf",
                        }
                    ]
                })
                
                response = client.chat.completions.create(
                    model=llm_type,
                    messages=messages,
                )
                result_text = response.choices[0].message.content
                is_valid, result_dict = self.result_text_check(
                    result_text, entry, check_target=check_target
                )
                
                if is_valid:
                    entry['is_reranked'] = True
                    entry['result_text'] = result_dict['result_text']
                    # entry['reasoning_text'] = result_dict['reasoning_text']
                    entry['reranked_dict'] = result_dict['reranked_dict']
                    entry['reranked_smiles'] = result_dict['reranked_smiles']
                    break
                    
            if not is_valid:
                entry['is_reranked'] = False
                entry['result_text'] = result_dict['result_text']
                # entry['reasoning_text'] = result_dict['reasoning_text']
                entry['reranked_dict'] = result_dict['reranked_dict']
                entry['reranked_smiles'] = result_dict['reranked_smiles']
                
        except Exception as e:
            print(f"Error processing entry idx {entry['idx']}: {e}")
            traceback.print_exc()
            entry['is_reranked'] = False
            entry['result_text'] = ''
            # entry['reasoning_text'] = ''
            entry['reranked_dict'] = {}
            entry['reranked_smiles'] = []
        finally:
            if api_token:
                self._return_token(api_token)

        return entry


    def llm_rerank_multiprocess(self, 
                                dataset: list, 
                                llm_type: str,
                                trials: int,
                                check_target: bool,
                                use_history_conversation: bool,
                                history_conversation_num: int) -> List[Dict[str, Any]]:
        """
        Rerank candidate molecules in the dataset using LLM with multiprocessing.
        
        Args:
            dataset: Dataset containing molecular entries
            prompt_path: Path to the prompt template file
            llm_type: Type of LLM model to use
            reasoning: Whether to enable reasoning mode
            candidates_info_idx: Optional index for candidate information
            chunk_size: Chunk size for processing (for future optimization)
            
        Returns:
            List of processed entries with reranking results
        """
        def process_single(entry: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single entry."""
            return self.llm_rerank(
                entry, llm_type, trials, check_target, use_history_conversation, history_conversation_num
            )
            
        self._validate_llm_type(llm_type)
        
        results = []
        
        # Process with ThreadPoolExecutor and progress bar
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                # Use tqdm for progress tracking
                with tqdm(
                    total=len(dataset), 
                    desc=f"Reranking with {llm_type}",
                    unit="entries"
                ) as pbar:
                    
                    # Process entries and collect results
                    for result in executor.map(process_single, dataset):
                        results.append(result)
                        pbar.update(1)
                        
                        # Optional: Update progress description with success/failure counts
                        success_count = sum(1 for r in results if r.get('is_reranked', False))
                        pbar.set_postfix({
                            'Success': success_count,
                            'Failed': len(results) - success_count
                        })
                        
            except KeyboardInterrupt:
                executor.shutdown(wait=False)
                raise
            except Exception as e:
                raise e
        
        return results
