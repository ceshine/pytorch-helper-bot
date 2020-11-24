import weakref
from functools import wraps
from typing import Sequence

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR as CosineAnnealingLR_
from torch.optim import Optimizer

__all__ = [
    "BaseLRScheduler", "LinearLR",
    "ExponentialLR", "TriangularLR",
    "GradualWarmupScheduler", "MultiStageScheduler",
    "CosineAnnealingScheduler"
]


class BaseLRScheduler(_LRScheduler):
    def switch_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer._step_count = self._step_count

    def clear_optimizer(self):
        self.optimizer = None


class CosineAnnealingScheduler(CosineAnnealingLR_, BaseLRScheduler):
    pass


class LinearLR(BaseLRScheduler):
    """Linearly increases or decrease the learning rate between two boundaries over a number of
    iterations.
    """

    def __init__(self, optimizer, min_lr_ratio, total_epochs, upward=True, last_epoch=-1):
        """Initialize a scheduler.

        Parameters
        ----------
        optimizer : Union[torch.optim.Optimizer, apex.fp16_utils.fp16_optimizer.FP16_Optimizer]
        min_lr_ratio : float
            min_lr_ratio * base_lr will be the starting learning rate.
        total_epochs : int
            the total number of "steps" in this run.
        last_epoch : int, optional
            the index of last epoch, by default -1.
        """
        assert min_lr_ratio < 1
        self.upward = upward
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs - 1  # starts at zero
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch
        if self.upward:
            progress = 1 - current_epoch / self.total_epochs  # 1 to 0
        else:
            progress = current_epoch / self.total_epochs  # 1 to 0
        # safety measure
        progress = max(min(progress, 1.), 0.)
        return [
            base_lr - progress * (base_lr - self.min_lr_ratio * base_lr)
            for base_lr in self.base_lrs
        ]


class ExponentialLR(BaseLRScheduler):
    """Exponentially increases the learning rate between two boundaries over
    a number of iterations.

    Mainly used by LR finders.
    """

    def __init__(self, optimizer, min_lr_ratio, total_epochs, last_epoch=-1):
        """Initialize a scheduler.

        Parameters
        ----------
        optimizer : Union[torch.optim.Optimizer, apex.fp16_utils.fp16_optimizer.FP16_Optimizer]
        min_lr_ratio : float
            min_lr_ratio * base_lr will be the starting learning rate.
        total_epochs : int
            the total number of "steps" in this run.
        last_epoch : int, optional
            the index of last epoch, by default -1.
        """
        assert min_lr_ratio < 1
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs - 1  # start from zero
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        progress = 1 - current_epoch / self.total_epochs  # 1 to 0
        return [base_lr * (self.min_lr_ratio) ** progress for base_lr in self.base_lrs]


class TriangularLR(BaseLRScheduler):
    def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        self.history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        new_lr = [
            lr * (1 + multiplier) / (self.max_mul + 1) for lr in self.base_lrs]
        self.history.append(new_lr)
        return new_lr


class GradualWarmupScheduler(BaseLRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Source: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epochs, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epochs = total_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epochs + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

    def switch_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer._step_count = self._step_count
        if self.after_scheduler:
            self.after_scheduler.optimizer = optimizer
            self.after_scheduler.optimizer._step_count = self.after_scheduler._step_count

    def clear_optimizer(self):
        self.optimizer = None
        if self.after_scheduler:
            self.after_scheduler.optimizer = None


class MultiStageScheduler:
    def __init__(self, schedulers: Sequence, start_at_epochs: Sequence[int], last_epoch: int = -1):
        assert len(schedulers) == len(start_at_epochs)
        schedulers, start_at_epochs = (
            np.array(schedulers), np.array(start_at_epochs))
        # sort starting epochs in descending order
        idx = np.flip(np.argsort(start_at_epochs))
        self.schedulers = schedulers[idx]
        self.start_at_epochs = start_at_epochs[idx]
        self.last_epoch = last_epoch
        # Explicitly run step(). Otherwise the initial LR will be initialized by the last sub-scheduler
        self.step(0)

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch - 1
        for scheduler, starting_epoch in zip(self.schedulers, self.start_at_epochs):
            if self.last_epoch +1 >= starting_epoch:
                scheduler.last_epoch = self.last_epoch - starting_epoch
                return scheduler.step()

    def switch_optimizer(self, optimizer):
        for scheduler in self.schedulers:
            scheduler.optimizer = optimizer
            scheduler.optimizer._step_count = scheduler._step_count

    def clear_optimizer(self):
        for scheduler in self.schedulers:
            scheduler.optimizer = None
