from unicore import options, tasks
import torch
from datasets import Dataset
from nmr_rise.utils.data import conformation_construction
from nmr_rise.mol2nmr.inference.infer import load_checkpoint_to_cpu
from unicore.data import (TokenizeDataset, PrependTokenDataset, AppendTokenDataset, NestedDictionaryDataset, RightPadDataset)
from nmr_rise.mol2nmr.data import (
    HuggingFaceDataset,
    KeyDataset, 
    IndexDataset, 
    ConformerSampleDataset, 
    TTADataset, 
    TTAIndexDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    SelectTokenDataset,
    ToTorchDataset,
    LatticeMatrixNormalizeDataset,
    GlobalDistanceDataset,
    PrependAndAppend3DDataset,
    RightPadDataset3D,
    DistanceDataset,
    PrependAndAppend2DDataset,
    RightPadDataset2D,
    RightPadDataset2D0,
    EdgeTypeDataset,
    TargetScalerDataset,
)
from tqdm import tqdm

def args_load(args_dict):
    save_dir = args_dict['save_dir']
    weight_name = args_dict['weight_name']
    data_path = args_dict['data_path']
    batch_size = args_dict['batch_size']
    task = args_dict['task']
    loss = args_dict['loss']
    arch = args_dict['arch']
    dict_name = args_dict['dict_name']
    # ckpt_path = args_dict['ckpt_path']
    selected_atom = args_dict['selected_atom']
    atom_des = args_dict['atom_des']
    split_mode = args_dict['split_mode']
    # Simulate command-line arguments
    args_list = [
        '--user-dir', None,
        data_path,
        '--valid-subset', 'valid',
        '--results-path', save_dir,
        '--saved-dir', save_dir,
        '--num-workers', '0',
        '--ddp-backend=c10d',
        '--batch-size', str(batch_size),
        '--task', task,
        '--loss', loss,
        '--arch', arch,
        '--dict-name', f"{dict_name}.txt",
        '--path', f'{save_dir}/{weight_name}.pt',
        '--fp16',
        '--fp16-init-scale', '4',
        '--fp16-scale-window', '256',
        '--log-interval', '50',
        '--log-format', 'simple',
        '--required-batch-size-multiple', '1',
        '--selected-atom', selected_atom,
        '--atom-descriptor', atom_des,
        '--split-mode', split_mode,
        '--gaussian-kernel',
        '--atom-descriptor', atom_des,
        '--split-mode', split_mode
    ]

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, args_list)
    return args

class mol2nmr_inference:
    def __init__(self, args_dict):
        self.args = args_load(args_dict)
        state = load_checkpoint_to_cpu(self.args.path)
        self.task = tasks.setup_task(self.args)
        self.model = self.task.build_model(self.args)
        self.model.load_state_dict(state['model'], strict=True)
        use_fp16 = self.args.fp16
        use_cuda = torch.cuda.is_available() and not self.args.cpu
        if use_fp16:
            self.model.half()
        if use_cuda:
            self.model.cuda()
        self.loss = self.task.build_loss(self.args)
        self.loss.eval()

    def process_dataset_for_mol2nmr(self, dataset: HuggingFaceDataset):
        assert isinstance(dataset, HuggingFaceDataset)
        if self.task.args.has_matid:
            matid_dataset = KeyDataset(dataset, "matid")
        else:
            matid_dataset = IndexDataset(dataset)
        # Do not run actually
        if self.task.args.conformer_augmentation:
            dataset = TTADataset(dataset, self.task.seed, "atoms", "coordinates_list", self.task.args.conf_size)
            matid_dataset = TTAIndexDataset(matid_dataset, self.task.args.conf_size)
        # Do not run actually
        if self.task.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        # Delete atom coordinates randomly if the number of atoms exceeds max_atoms
        dataset = CroppingDataset(dataset, self.task.seed, "atoms", "coordinates", self.task.args.max_atoms)
        # Normalization coordinates = coordinates - coordinates.mean(axis=0)
        dataset = NormalizeDataset(dataset, "coordinates")

        # Create dataset for token, selected atoms, coordinates
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.task.dictionary, max_seq_len=self.task.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.task.selected_token)
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # Append BOS and EOS tokens to token_dataset and correspondingly pad select_atom_dataset
        token_dataset = PrependAndAppend(token_dataset, self.task.dictionary.bos(), self.task.dictionary.eos())
        select_atom_dataset = PrependAndAppend(select_atom_dataset, self.task.dictionary.pad(), self.task.dictionary.pad())

        # Convert dataset to torch tensor from numpy array
        coord_dataset = ToTorchDataset(coord_dataset, 'float32')

        if self.task.args.global_distance:
            lattice_matrix_dataset = LatticeMatrixNormalizeDataset(dataset, 'lattice_matrix')
            # logger.info("use global distance: {}".format(task.args.global_distance))
            distance_dataset = GlobalDistanceDataset(coord_dataset, lattice_matrix_dataset)
            distance_dataset = PrependAndAppend3DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset3D(distance_dataset, pad_idx=0)
        else:
            # Compute pairwise distance matrix from coordinates
            distance_dataset = DistanceDataset(coord_dataset)
            # Initial Padding
            distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
            # Batch Padding.
            distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        edge_type = EdgeTypeDataset(token_dataset, len(self.task.dictionary))

        tgt_dataset = KeyDataset(dataset, "atoms_target")
        # Normalization
        tgt_dataset = TargetScalerDataset(tgt_dataset, self.task.target_scaler, self.task.args.num_classes)
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')

        tgt_dataset = PrependAndAppend(tgt_dataset, self.task.dictionary.pad(), self.task.dictionary.pad())

        if self.task.args.atom_descriptor != 0:
            atomdes_dataset = KeyDataset(dataset, "atoms_descriptor")
            atomdes_dataset = ToTorchDataset(atomdes_dataset, dtype='float32')
            atomdes_dataset = PrependAndAppend(atomdes_dataset, self.dictionary.pad(), self.dictionary.pad())
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                            "atom_descriptor": RightPadDataset2D0(
                                atomdes_dataset,
                                pad_idx=0,
                            ),
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                    },
                )
        else:
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.task.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.task.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                    },
                )
        return nest_dataset
    
    def infer_single_entry(self, smiles: str, show_progress = False):
        single_data = {'smiles': smiles}
        single_data = conformation_construction(single_data)
        dataset = Dataset.from_list([single_data])
        dataset = HuggingFaceDataset(dataset=dataset)
        return self.infer_dataset(dataset, show_progress = show_progress)

    def infer_batch_entry(self, smiles_list: list, show_progress = False):
        batch_data = [{'smiles': smiles} for smiles in smiles_list]
        batch_data = [conformation_construction(entry) for entry in batch_data]
        dataset = Dataset.from_list(batch_data)
        dataset = HuggingFaceDataset(dataset=dataset)
        return self.infer_dataset(dataset, show_progress = show_progress)

    def infer_dataset(self, dataset, show_progress = False):
        if type(dataset) is HuggingFaceDataset:
            processed_dataset = self.process_dataset_for_mol2nmr(dataset)
        elif type(dataset) is Dataset:
            processed_dataset = self.process_dataset_for_mol2nmr(HuggingFaceDataset(dataset=dataset))
        else:
            raise ValueError("dataset must be of type HuggingFaceDataset or Dataset")
        itr = self.task.get_batch_iterator(
            dataset=processed_dataset,
            batch_size=self.args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=1,
            shard_id=0,
            num_workers=self.args.num_workers,
            data_buffer_size=self.args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        dataset_with_predict = []
        for _, sample in tqdm(enumerate(itr), total=len(itr), disable=not show_progress):
            sample_size = len(sample['matid'])
            from unicore import utils
            sample = utils.move_to_cuda(sample)
            _, _, log_output = self.task.valid_step(sample, self.model, self.loss, test=True)
            import numpy as np
            select_atom_num_list = log_output['select_atom'].reshape(sample_size, -1).sum(dim=1).cpu().numpy().tolist()
            accumu_num = np.cumsum([0] + select_atom_num_list).tolist()
            predict_list = [log_output['predict'][accumu_num[i-1]:accumu_num[i]].squeeze().tolist() for i in range(1, len(accumu_num))]

            select_atom = log_output['select_atom'].reshape(sample_size,-1)
            for i in range(sample_size):
                try:
                    predict_peaks = predict_list[i] if isinstance(predict_list[i], list) else [predict_list[i]]
                    data_idx = log_output['matid'][i].item()
                    atom_target_mask = dataset[data_idx]['atoms_target_mask']
                    _select_atom = select_atom[i][1:1+len(atom_target_mask)]
                    predict = []
                    flag = 0
                    for j in range(len(_select_atom)):
                        if _select_atom[j]:
                            predict.append(predict_peaks[flag])
                            flag += 1
                        else:
                            predict.append(0.0)
                    dataset_with_predict.append({**dataset[data_idx], 'predict': np.array(predict)})
                except Exception as e:
                    print(f"Error processing data_idx {data_idx}: {e}")
                    print(sample['matid'][i])
        return Dataset.from_list(dataset_with_predict)

