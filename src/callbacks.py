import os
import stat
import time
import numpy as np
import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
        Monitor the loss in training.
        If the loss is NAN or INF, terminating training.
    """

    def __init__(self, epoch_size, logger, per_print_time=1):
        super(LossCallBack, self).__init__()
        self.epoch_size = epoch_size
        self.logger = logger
        self.per_print_time = per_print_time
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

    def _get_optimizer_from_cbp(self, cb_params):
        dataset_sink_mode = cb_params.get('dataset_sink_mode', True)
        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        elif dataset_sink_mode:
            optimizer = cb_params.train_network.network.optimizer
        else:
            optimizer = cb_params.train_network.optimizer
        return optimizer

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        output = cb_params.net_outputs
        cond = None
        loss_scale = None
        data_sink_mode = cb_params.get('dataset_sink_mode', True)
        if not data_sink_mode:
            if isinstance(output, (tuple, list)):
                if isinstance(output[0], ms.Tensor) and isinstance(output[0].asnumpy(), np.ndarray):
                    loss = output[0]
                    cond = output[1]
                    loss_scale = output[2]
            if isinstance(output, ms.Tensor) and isinstance(output.asnumpy(), np.ndarray):
                loss = np.mean(output.asnumpy())

            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            cur_epoch_num = cb_params.cur_epoch_num
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num, cur_step_in_epoch))

            if cur_step_in_epoch % self.per_print_time == 0:
                opt = self._get_optimizer_from_cbp(cb_params)
                cur_global_step = opt.global_step
                cur_lr = opt.learning_rate(cur_global_step - 1)
                per_step_time = 1000 * (time.time() - self.step_start_time) / self.per_print_time
                log_info = "epoch: [%s/%s] step: [%s/%s], cur global step: %s, lr: %.6f, loss: %.6f, overflow: %s, " \
                           "loss_scale: %s, per step time: %.3f ms" % (
                               cur_epoch_num, self.epoch_size, cur_step_in_epoch, cb_params.batch_num,
                               cur_global_step.asnumpy().item(),
                               cur_lr, loss, cond, loss_scale, per_step_time)
                self.logger.info(log_info)
                self.step_start_time = time.time()

        def on_train_epoch_begin(self, run_context):
            self.epoch_start_time = time.time()
            self.step_start_time = time.time()

        def on_train_epoch_end(self, run_context):
            cb_params = run_context.original_args()
            loss = cb_params.net_outputs
            cur_epoch_num = cb_params.cur_epoch_num
            if isinstance(loss, (tuple, list)):
                if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                    loss = loss[0]
            if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
                loss = np.mean(loss.asnumpy())

            epoch_time = time.time() - self.epoch_start_time
            log_info = "epoch: [%s/%s], loss: %.6f, epoch time: %.3f s, per step time: %.3f ms" % (
                cur_epoch_num, self.epoch_size, loss, epoch_time, epoch_time * 1000 / cb_params.batch_num
            )
            self.logger.info(log_info)
