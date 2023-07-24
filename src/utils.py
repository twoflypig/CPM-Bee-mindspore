import math
import time
import numpy as np
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, dataset_size=-1, local_rank=0, has_trained_epoch=0, has_trained_step=0, micro_size=1,
                 is_last_stage=True):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        self.micro_size = micro_size
        self.is_last_stage = is_last_stage
        print("Load the trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            loss_value = 'no loss for this stage'
            if self.is_last_stage:
                loss_value = cb_params.net_outputs[0].asnumpy() / self.micro_size
                loss_value = np.mean(loss_value)
            print("time: {} local_rank: {}, epoch: {}, step: {}, loss is {}, overflow is {}, loss scale is {}".
                  format(date, int(self.local_rank), int(epoch_num) + int(self.has_trained_epoch),
                         cb_params.cur_step_num + int(self.has_trained_step), loss_value,
                         cb_params.net_outputs[1].asnumpy(), cb_params.net_outputs[2].asnumpy()))
