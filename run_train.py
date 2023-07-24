import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore import Tensor, nn
from mindspore.communication import init

from src.models import CPMBeeConfig, BeeForward


def get_dataset(batch, seqlen, num_segment_bucket, ext_table_size, step_per_epoch):
    """
    """
    input = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    label = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    input_sub = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    length = np.random.randint(1, seqlen, (batch,)).astype(np.int32)
    context = np.full((batch, seqlen), 1).astype(np.bool_)
    sample_ids = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    num_segments = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    segment = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel_offset = np.random.randint(0, 1, (batch, seqlen)).astype(np.int32)
    segment_rel = np.random.randint(0, 1, (batch, num_segment_bucket)).astype(np.int32)
    span = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    ext_table_ids = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    ext_table_sub = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)

    def generate():
        for _ in range(step_per_epoch):
            yield Tensor(input), Tensor(input_sub), Tensor(length), Tensor(context), Tensor(sample_ids), \
                  Tensor(num_segments), Tensor(segment), Tensor(segment_rel), Tensor(segment_rel_offset), \
                  Tensor(span), Tensor(ext_table_ids), Tensor(ext_table_sub), Tensor(label)

    return generate


def get_simple_dataset(batch, seqlen, ext_table_size, step_per_epoch):
    """
    """
    input = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    label = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    input_sub = np.random.randint(0, 1000, (batch, seqlen)).astype(np.int32)
    position = np.random.randint(0, seqlen, (batch, seqlen)).astype(np.int32)
    segment_bucket = np.full((batch, seqlen, seqlen), 1).astype(np.int32)
    attention_mask = np.full((batch, seqlen, seqlen), 1).astype(np.bool_)
    ext_table_ids = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)
    ext_table_sub = np.random.randint(1, ext_table_size, (ext_table_size,)).astype(np.int32)

    def generate():
        for _ in range(step_per_epoch):
            yield Tensor(input), Tensor(input_sub), Tensor(position), Tensor(segment_bucket), \
                Tensor(attention_mask), Tensor(ext_table_ids), Tensor(ext_table_sub), Tensor(label)

    return generate

cpm_2b_config = {
    "vocab_size": 86583,
    "dim_model": 4096,
    "dim_ff" : 5120,
    "num_layers" : 48,
    "num_heads": 32,
    "dim_head" : 64,
    "dropout_p" : 0.0,
    "position_bias_num_buckets" : 256,
    "position_bias_num_segment_buckets": 256,
    "position_bias_max_distance" : 2048,
    "eps" : 1e-6,
    "half" : True,
    "mask_modules": [[False, False], [True, False], [False, False], [True, False], [True, True], [True, False], [True, True], [True, True], [False, False], [False, False], [True, True], [True, False], [True, False], [True, True], [False, False], [True, True], [False, False], [False, True], [True, False], [True, True], [False, False], [False, True], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [False, False], [True, True], [True, False], [True, True], [True, True], [True, True], [False, False], [True, True], [False, False], [True, True], [True, True], [False, False], [True, True], [False, False], [False, False]]
}


if __name__ == '__main__':
    pass
