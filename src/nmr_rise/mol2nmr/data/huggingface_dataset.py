from datasets import Dataset, load_from_disk
from pyparsing import lru_cache
import numpy as np

class HuggingFaceDataset(Dataset):
    def __init__(self, db_path = None, dataset = None, split='train'):
        if dataset is not None:
            self.dataset = dataset
        elif db_path is not None:
            self.dataset = load_from_disk(db_path)[split]
        else:
            raise ValueError("Either db_path or dataset must be provided")
        column_list = ['atoms', 'coordinates', 'atoms_target', 'atoms_target_mask', 'smiles_with_hydrogens', 'db_id', 'inchikey']
        self.dataset = self.dataset.remove_columns([col for col in self.dataset.column_names if col not in column_list])
        # rename smiles_with_hydrogens to smiles
        self.dataset = self.dataset.rename_column('smiles_with_hydrogens', 'smiles')
        

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[int(idx)]
        data['coordinates'] = np.array(data['coordinates'])
        data['atoms_target_mask'] = np.array(data['atoms_target_mask'])
        data['atoms_target'] = np.array(data['atoms_target'])
        return data