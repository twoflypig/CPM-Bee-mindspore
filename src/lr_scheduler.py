import mindspore
from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

import mindspore.common.dtype as mstype


class Noam(LearningRateSchedule):
    def __init__(self, start_lr, warmup_iter, end_iter):
        super().__init__()
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter

    def get_lr_warmup(self, num_iter):
        return self.start_lr / ops.sqrt(
            ops.scalar_to_tensor(self.warmup_iter)) * num_iter.to(mstype.float32) / self.warmup_iter

    def get_lr_decay(self, num_iter):
        return self.start_lr / ops.sqrt(num_iter.to(mstype.float32))

    def construct(self, global_step):
        if global_step + 1 < self.warmup_iter:
            cur_lr = self.get_lr_warmup(global_step + 1)
        else:
            cur_lr =  self.get_lr_decay(global_step + 1)
        return cur_lr
