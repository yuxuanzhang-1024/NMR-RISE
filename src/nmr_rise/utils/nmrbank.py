from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import random
import ast
import numpy as np
import pandas as pd

# Process H-NMR sequence of NMRBank dataset to a unified format using LLM
class LLMThreadProcessor:
    def __init__(self, api_tokens: List[str], max_workers: int = None):
        """
        Initialize the thread processor with multiple DeepSeek API tokens
        """
        self.api_tokens = api_tokens
        self.max_workers = max_workers or len(api_tokens)
        self.available_tokens = queue.Queue()
        
        # Fill the token queue
        for token in api_tokens:
            self.available_tokens.put(token)
    
    def llm_hnmr_extraction(self, hnmr_sequence: str) -> str:
        """
        Process a single HNMR sequence using available token
        """
        # Get an available token
        api_token = self.available_tokens.get()
        
        try:
            # Read prompt file
            with open('./prompt/hnmr_process.txt', 'r') as file:
                prompt = file.read()
            
            # Create OpenAI client with DeepSeek API
            client = OpenAI(
                api_key=api_token, 
                base_url="https://api.deepseek.com"
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a chemist specialized in NMR spectrum analysis."},
                    {"role": "user", "content": prompt.format(hnmr_sequence=hnmr_sequence)},
                ],
            )
            
            return response.choices[0].message.content
            
        finally:
            # Put the token back for reuse
            self.available_tokens.put(api_token)
    
    def process_multiple_sequences_threaded(self, hnmr_sequences: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process multiple HNMR sequences using threading
        """
        def process_single(entry):
            try:
                result = self.llm_hnmr_extraction(entry['hnmr_sequence'])
                return {
                    'PMID': entry['PMID'],
                    'sequence': entry['hnmr_sequence'],
                    'result': result,
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'PMID': entry['PMID'],
                    'sequence': entry['hnmr_sequence'],
                    'result': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single, hnmr_sequences))
        
        return results
    
def nmrbank_process(file_path: str):
    NMRBank = pd.read_csv(file_path)
    new_data_list = []
    error_list = []
    filter_list = []
    for i in range(len(data_list)):
        try:
            smiles = NMRBank.iloc[i]['Standardized SMILES']
            mol = Chem.MolFromSmiles(smiles)
            molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            h_nmr_peaks = []
            c_nmr_peaks = []
            cnmr = NMRBank.iloc[i]['13C NMR chemical shifts']
            cnmr = [float(s.strip()) for s in cnmr.split(',') if s.strip()]
            for peak in cnmr:
                c_nmr_peaks.append({'delta (ppm)': peak, 'integral': None, 'intensity': None, 'width (ppm)': None})
        
            hnmr = ast.literal_eval(data_list[i]['result'].split('<answer_list>')[1].split('</answer_list>')[0].strip())
            for h in hnmr:
                if h['value'] == None:
                    if h['rangeMin'] != None and h['rangeMax'] != None:
                        h['value'] = random.uniform(h['rangeMin'], h['rangeMax'])
                    else:
                        continue
                if h['rangeMin'] == None or h['rangeMax'] == None:
                    if h['value'] != None:
                        h['rangeMin'] = h['value'] - random.uniform(0, 0.5) if h['rangeMin'] == None else h['rangeMin']
                        h['rangeMax'] = h['value'] + random.uniform(0, 0.5) if h['rangeMax'] == None else h['rangeMax']
                    else:
                        continue
                h_nmr_peaks.append({'category': h['type'], 'centroid': h['value'], 'delta': h['value'], 'j_values': None, 'nH': 1, 'rangeMax': h['rangeMax'], 'rangeMin': h['rangeMin']},)
            num_C = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            num_H = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            # if len(h_nmr_peaks) == num_H and len(c_nmr_peaks) == num_C:
            new_data_list.append({
                'idx': i,
                'molecular_formula': molecular_formula,
                'num_C': num_C,
                'num_H': num_H,
                'c_nmr_peaks': c_nmr_peaks,
                'h_nmr_peaks': h_nmr_peaks,
                'num_h_peaks': len(h_nmr_peaks),
                'num_c_peaks': len(c_nmr_peaks),
                'smiles': smiles,
                'ir_spectra': np.zeros((1800,),dtype=np.float32)
            })
            # else:
            #     filter_list.append(i)
        except:
            error_list.append(i)
