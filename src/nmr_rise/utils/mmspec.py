import os
from datasets import Dataset, load_dataset
from typing import Tuple

def split_dataset(
    input_dir: str,
    output_dir: str,
    columns: list[str],
    num_cpus: 1,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: int = 42
):
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    dataset = load_dataset('parquet', data_dir=input_dir, split='train', columns=columns, num_proc=num_cpus)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=random_state)
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])

    train_dataset = dataset.select(range(0, train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))

    # Save the dataset in parquet format (maximum 3000 rows per file)
    def save_dataset_in_chunks(ds: Dataset, split_name: str):
        chunk_size = 3000
        num_chunks = (len(ds) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            chunk = ds.select(range(i * chunk_size, min((i + 1) * chunk_size, len(ds))))
            chunk.to_parquet(os.path.join(output_dir, split_name, f'aligned_chunk_{i}.parquet'))

    save_dataset_in_chunks(train_dataset, 'train')
    save_dataset_in_chunks(val_dataset, 'valid')
    save_dataset_in_chunks(test_dataset, 'test')

split_dataset(
    input_dir='./data/pretrain/raw',
    output_dir='./data/pretrain/split',
    columns=['molecular_formula', 'h_nmr_peaks', 'c_nmr_peaks', 'smiles'],
    num_cpus=os.cpu_count(),
    split_ratio=(0.9, 0.05, 0.05),
    random_state=42
)