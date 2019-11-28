import math
from typing import Union, Sequence

import torch
from torch.optim import Optimizer

__all__ = ["WeightDecayOptimizerWrapper", "AdamW"]


class WeightDecayOptimizerWrapper(Optimizer):
    def __init__(self, optimizer: Optimizer, weight_decay: Union[Sequence[float], float], change_with_lr: bool = True) -> None:
        self.optimizer = optimizer
        if isinstance(weight_decay, (list, tuple)):
            assert len(weight_decay) == len(self.optimizer.param_groups)
            assert all((x >= 0 for x in weight_decay))
            self.weight_decays = weight_decay
        else:
            assert weight_decay >= 0
            self.weight_decays = [weight_decay] * \
                len(self.optimizer.param_groups)
        self.state = self.optimizer.state
        self.change_with_lr = change_with_lr

    def step(self, closure=None) -> None:
        for group, weight_decay in zip(self.optimizer.param_groups, self.weight_decays):
            for param in group['params']:
                if param.grad is None or weight_decay == 0:
                    continue
                if self.change_with_lr:
                    param.data.add_(
                        -weight_decay * group['lr'], param.data)
                else:
                    param.data.add_(-weight_decay, param.data)
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.weight_decays = state_dict["weight_decays"]
        self.change_with_lr = state_dict["change_with_lr"]
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self):
        return {
            'weight_decays': self.weight_decays,
            'change_with_lr':  self.change_with_lr,
            'optimizer': self.optimizer.state_dict()
        }

    def __repr__(self):
        return self.optimizer.__repr__()

    def __getstate__(self):
        return {
            'weight_decays': self.weight_decays,
            'change_with_lr':  self.change_with_lr,
            'optimizer': self.optimizer
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.state = self.optimizer.__getstate__()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Copied from huggingface/transformers Github repo.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * \
                        math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss
