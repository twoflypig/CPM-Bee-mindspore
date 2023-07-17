from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Constant


class LayerNorm(nn.Cell):
    """RMS LayerNorm"""

    def __init__(
            self,
            dim_norm: int,
            dtype: mstype.float_ = mstype.half,
            eps: float = 1e-6,
            init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = Parameter(initializer(Constant(init_var), (dim_norm,), dtype=dtype), 'weight')
        self.cast = ops.Cast()
        self.rsqrt = ops.Rsqrt()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.add = ops.Add()
        self.mul = ops.Mul()
        self.expand = ops.ExpandDims()

    def construct(self, x: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.shape[-1] == self.dim_norm
        old_dtype = x.dtype
        variance = self.cast(x, mstype.float32)
        variance = self.square(variance)
        variance = self.mean(variance, -1)
        x = (x * self.rsqrt(self.add(variance, self.eps))).astype(old_dtype)
        return self.mul(x, self.weight)

    def shard(self, dp, mp):
        self.cast.shard(((dp, mp, 1),))
        self.square.shard(((dp, mp, 1),))
        self.mean.shard(((dp, mp, 1),))
        self.add.shard(((dp, mp, 1),))
        self.expand.shard(((dp, mp, 1),))
        self.mul.shard(((dp, mp, 1),))