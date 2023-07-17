import multiprocessing
import numpy as np
import mindspore as ms
import mindspore.dataset as de
from src.tokenizers import CPMBeeTokenizer
from src.dataset import SimpleDataset
from src.models import CPMBee
from src.data_converter import _MixedDatasetSaver, _MixedDatasetConfig


class FineTuneDataset:
    def __init__(self, dataset_path, max_length=2048, max_depth=8, pad_len=7000, ext_pad_len=8):
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.max_depth = max_depth
        self.pad_length = pad_len
        self.ext_pad_length = ext_pad_len
        self.data_lst = self.create_dataset_list()

    def create_dataset_list(self):
        dataset = SimpleDataset(self.dataset_path, shuffle=False)
        _packer = _MixedDatasetSaver(
            1, self.max_length, self.tokenizer, self.max_depth
        )
        _ds_cfg: _MixedDatasetConfig = {
            "weight": 1.0,
            "path": self.dataset_path,
            "transforms": [],
            "task_name": "test_task",
            "dataset_name": "finetune",
            "incontext_weight": [1.0],
            "lines": len(dataset),
            "dataset": dataset,
        }
        data_lst = list()
        while True:
            try:
                batch = _packer.add_data(_ds_cfg)
            except EOFError:
                break
            if batch is None:
                continue
            else:
                break

            inputs = batch['inputs'][0],  # (batch, seqlen) int32
            inputs_sub = batch['inputs_sub'][0],  # (batch, seqlen) int32
            length = batch['length'][0],  # (batch) int32

            context = batch['context'][0],  # (batch, seqlen) bool
            sample_ids = batch['sample_ids'][0],  # (batch, seq_len) int32
            num_segments = batch['num_segments'][0],  # (batch, seq_len) int32

            segment_ids = batch['segment_ids'][0],  # (batch, seqlen) int32
            segment_rel_offset = batch['segment_rel_offset'][0],  # (batch, seq_len) int32

            segment_rel = batch['segment_rel'][0]
            segment_rel_ = np.zeros(self.pad_length, dtype=np.int32)
            segment_rel_[:segment_rel.shape[0]] = segment_rel

            # (batch, num_segment_bucket) int32
            spans = batch['spans'][0],  # (batch, seqlen) int32
            ext_ids = batch['ext_ids']
            ext_ids_ = np.zeros(self.ext_pad_length, dtype=np.int32)
            ext_ids_[:ext_ids.shape[0]] = ext_ids_
            # (ext_table_size) int32
            ext_sub = batch['ext_sub']
            ext_sub_ = np.zeros(self.ext_pad_length, dtype=np.int32)
            exe_sub_[:ext_sub.shape[0]] = ext_sub
            # (ext_table_size) int32
            target = batch['target'][0]

            data_lst.append((inputs, inputs_sub, length, context, sample_ids, num_segments, segment_ids,
                             segment_rel_offset, segment_rel, spans, ext_ids_, ext_sub_, target))
        return data_lst

    def __len__(self):
        return 944

    def __getitem__(self, item):
        return self.data_lst[item]


def create_cpm_finetune_dataset(dataset_path, max_length=2048, max_depth=8, pad_len=7000, ext_pad_len=8, batch_size=1,
                                num_shard=None, shard_id=None):
    cpm_dataset = FineTuneDataset(dataset_path, max_length, max_depth, pad_len, ext_pad_len)
    columns_name = ["input", "input_sub", "length", "context", "sample_ids", "num_segments", "segment", "segment_rel",
                    "segment_rel_offset", "span", "ext_table_ids", "ext_table_sub", "label"]
    cores = multiprocessing.cpu_count()
    device_num = num_shard if num_shard is not None else 1
    num_parallel_workers = int(cores / device_num)
    ds = de.GeneratorDataset(cpm_dataset, columns_name, num_parallel_workers=8,
                             python_multiprocessing=True, shard_id=shard_id, num_shards=num_shard)
    ds = ds.batch(batch_size, False)
    ds = ds.repeat(1)

    return ds
