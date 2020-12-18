import socket
from time import time
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Optional, Sequence
from pathlib import Path

import torch
import numpy as np
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

from .bot import BaseBot, StopTraining
from .cutmix_utils import cutmix_bbox_and_lam

__all__ = [
    "Callback", "MixUpCallback", "LearningRateSchedulerCallback",
    "StepwiseLinearPropertySchedulerCallback", "MovingAverageStatsTrackerCallback",
    "CheckpointCallback", "EarlyStoppingCallback", "TelegramCallback",
    "WandbCallback", "CutMixCallback", "RandomCallbackChoices"
]


class Callback:
    def on_batch_inputs(self, bot: BaseBot, input_tensors: torch.Tensor, targets: torch.Tensor, is_eval: bool):
        return input_tensors, targets

    def on_train_starts(self, bot: BaseBot):
        return

    def on_train_ends(self, bot: BaseBot):
        return

    def on_epoch_ends(self, bot: BaseBot, epoch: int):
        return

    def on_eval_starts(self, bot: BaseBot):
        return

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        return

    def on_step_ends(self, bot: BaseBot, train_loss: float, train_weight: int):
        return

    def on_load_checkpoint(self, **kwargs):
        return

    def on_save_checkpoint(self):
        return

    def reset(self):
        return


class RandomCallbackChoices:
    def __init__(self, callbacks: Sequence[Callback], p: Sequence[Callback]):
        self.p = np.asarray(p) / np.sum(p)
        self.callbacks = callbacks
        assert len(p) == len(callbacks)

    def get_callback(self):
        return np.random.choice(self.callbacks, p=self.p)

    def on_batch_inputs(self, bot: BaseBot, input_tensors: torch.Tensor, targets: torch.Tensor, is_eval: bool):
        return self.get_callback().on_batch_inputs(
            bot, input_tensors, targets, is_eval)

    def on_train_starts(self, bot: BaseBot):
        return self.get_callback().on_train_starts(bot)

    def on_train_ends(self, bot: BaseBot):
        return self.get_callback().on_train_ends(bot)

    def on_epoch_ends(self, bot: BaseBot, epoch: int):
        return self.get_callback().on_epoch_ends(bot, epoch)

    def on_eval_starts(self, bot: BaseBot):
        return self.get_callback().on_eval_starts(bot)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        return self.get_callback().on_eval_ends(bot, metrics)

    def on_step_ends(self, bot: BaseBot, train_loss: float, train_weight: int):
        return self.get_callback().on_step_ends(bot, train_loss, train_weight)

    def on_load_checkpoint(self, **kwargs):
        # Not Supported
        return

    def on_save_checkpoint(self):
        # Not Supported
        return

    def reset(self):
        # Not Supported
        return


class WandbCallback(Callback):
    """ Callback for the Weights and Biases service

    WARNING: Resuming is not fully supported yet.

    Reference: https://github.com/wandb/client/raw/ef0911c47beebab0db8749d764802057d3480e69/wandb/fastai/__init__.py
    """

    def __init__(
            self, config: Dict, name: str, run_name: Optional[str] = None,
            watch_level: Optional[str] = None, watch_freq: int = 100, log_freq: int = 2):
        if WANDB is False:
            raise ImportError(
                "Please install 'wandb' before using WandbCallback.")
        # project name can only be in lower case
        wandb.init(config=config, project=name.lower(), name=run_name)
        self.watch_level = watch_level
        self.watch_freq = watch_freq
        self.log_freq = log_freq
        self.project = name.lower()
        self.config = config
        self.project_name = name.lower()
        self.run_name = run_name

    def on_train_starts(self, bot: BaseBot):
        wandb.watch(bot.model, log=self.watch_level,
                    log_freq=self.watch_freq)

    def on_step_ends(self, bot: BaseBot, train_loss: float, train_weight: int):
        if bot.step % self.log_freq == 0:
            wandb.log({"train_loss": train_loss}, step=bot.step)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        metrics_ = {
            metric_name: metric_value
            for metric_name, (metric_value, _) in metrics.items()
        }
        # Rename to avoid conflicts
        metrics_["val_loss"] = metrics_["loss"]
        del metrics_["loss"]
        # NOTE: remember to train one more step to sync the final eval metrics to the server
        wandb.log(metrics_, step=bot.step)

    def log_summary(self, key, value):
        wandb.run.summary[key] = value

    def on_load_checkpoint(self, cold_start: bool, **kwargs):
        if cold_start:
            wandb.init(config=self.config, project=self.project_name, name=self.run_name)


class TelegramCallback(Callback):
    """A Telegram notification callback

    Reference: https://github.com/huggingface/knockknock
    """
    DATE_FORMAT = "%Y-%m-%d %H:%M:%d"

    def __init__(self, token: str, chat_id: int, name: str, report_evals: bool = False):
        try:
            import telegram
        except ImportError:
            raise ImportError(
                "Please install 'python-telegram-bot' before using TelegramCallback.")
        self._token = token
        self.telegram_bot = telegram.Bot(token=self._token)
        self.host_name = socket.gethostname()
        self.report_evals = report_evals
        self.chat_id = chat_id
        self.name = name
        self.start_time = None

    def on_train_starts(self, bot: BaseBot):
        self.start_time = datetime.now()
        contents = [
            f'{self.name} has started training 🎬',
            'Machine name: %s' % self.host_name,
            'Starting date: %s' % self.start_time.strftime(
                TelegramCallback.DATE_FORMAT)
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def on_train_ends(self, bot: BaseBot):
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        contents = [
            f'{self.name} has finished training 🎉',
            'Machine name: %s' % self.host_name,
            'Starting date: %s' % self.start_time.strftime(
                TelegramCallback.DATE_FORMAT),
            'End date: %s' % end_time.strftime(
                TelegramCallback.DATE_FORMAT),
            'Training duration: %s' % str(elapsed_time)
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        if self.report_evals is False:
            return
        contents = [
            f"Metrics from {self.name} at step {bot.step}:"
        ]
        contents += [
            f"{metric_name}: {metric_string}"
            for metric_name, (metric_value, metric_string) in metrics.items()
        ]
        text = '\n'.join(contents)
        self.telegram_bot.send_message(chat_id=self.chat_id, text=text)

    def on_load_checkpoint(self, **kwargs):
        import telegram
        self.telegram_bot = telegram.Bot(token=self._token)

    def on_save_checkpoint(self):
        self.telegram_bot = None


class CutMixCallback(Callback):
    """Assumes the first dimension is batch.

    Reference: https://github.com/rwightman/pytorch-image-models/blob/8c9814e3f500e8b37aae86dd4db10aba2c295bd2/timm/data/mixup.py
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False, minmax: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target
        self.minmax = minmax

    def on_batch_inputs(self, bot: BaseBot, input_tensors, targets, is_eval: bool):
        if is_eval is True:
            return input_tensors, targets
        batch = input_tensors[0]
        batch_flipped = batch.flip(0).clone()
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        for i in range(batch.shape[0]):
            (yl, yh, xl, xh), lambd_tmp = cutmix_bbox_and_lam(
                batch.shape, lambd[i], ratio_minmax=self.minmax, correct_lam=True)
            lambd[i] = lambd_tmp
            # fill in the cut regions
            batch[i, :, yl:yh, xl:xh] = batch_flipped[i, :, yl:yh, xl:xh]
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets.flip(0).float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets.flip(0) * (1-lambd_tensor)
            )
        # input_tensors[0] = batch
        return input_tensors, new_targets


class MixUpCallback(Callback):
    """Assumes the first dimension is batch.

    Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target

    def on_batch_inputs(self, bot: BaseBot, input_tensors, targets, is_eval: bool):
        if is_eval is True:
            return input_tensors, targets
        batch = input_tensors[0]
        permuted_idx = torch.randperm(batch.size(0)).to(batch.device)
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        lambd = np.concatenate(
            [lambd[:, np.newaxis], 1-lambd[:, np.newaxis]], axis=1
        ).max(axis=1)
        # Create the tensor and expand (for batch inputs)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(batch.size())-1)]
        ).expand(-1, *batch.shape[1:])
        # Combine input batch
        new_batch = (batch * lambd_tensor +
                     batch[permuted_idx] * (1-lambd_tensor))
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets[permuted_idx].float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets[permuted_idx] * (1-lambd_tensor)
            )
        input_tensors[0] = new_batch
        return input_tensors, new_targets


class LearningRateSchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        self.scheduler.step()

    def on_load_checkpoint(self, **kwargs):
        self.scheduler.switch_optimizer(kwargs["optimizer"])

    def on_save_checkpoint(self):
        self.scheduler.clear_optimizer()


class StepwiseLinearPropertySchedulerCallback(Callback):
    def __init__(self, target_obj, property_name, start_val, end_val, decay_start_step, decay_stop_step, log_freq):
        super().__init__()
        self.target_obj = target_obj
        self.property_name = property_name
        self.start_val = start_val
        self.end_val = end_val
        self.decay_start_step = decay_start_step
        self.decay_stop_step = decay_stop_step
        self.log_freq = log_freq

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        if bot.step % self.log_freq == 0:
            bot.logger.info(
                "%s %s %.4f",
                self.target_obj.__class__.__name__,
                self.property_name,
                getattr(self.target_obj, self.property_name))
        new_val = self.get_value(bot)
        setattr(self.target_obj, self.property_name, new_val)

    def get_value(self, bot):
        if self.start_val == self.end_val or bot.step <= self.decay_start_step:
            return self.start_val
        elif bot.step >= self.decay_stop_step:
            return self.end_val
        change = (self.end_val - self.start_val) * (
            (bot.step - self.decay_start_step) /
            (self.decay_stop_step - self.decay_start_step)
        )
        return self.start_val + change


class MovingAverageStatsTrackerCallback(Callback):
    """Keep moving average for training losses.

    Raw values for evaluation stats.
    """

    def __init__(self, avg_window: int, log_interval: int):
        super().__init__()
        self.avg_window = avg_window
        self.log_interval = log_interval
        self.reset()

    def on_train_starts(self, bot: BaseBot):
        self.timer = time()

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        if np.isnan(train_loss):
            # skip
            return
        self.train_losses.append(train_loss)
        self.train_weights.append(train_weight)
        if bot.step % self.log_interval == 0:
            train_loss_avg = np.average(
                self.train_losses, weights=self.train_weights)
            speed = (time() - self.timer) / self.log_interval
            # reset timer
            self.timer = time()
            bot.logger.info(
                f"Step %5d | loss {bot.loss_format} | lr: %.2e | %.3fs per step",
                bot.step, train_loss_avg, bot.optimizer.param_groups[-1]['lr'],
                speed
            )
            bot.logger.tb_scalars(
                "lr", bot.optimizer.param_groups[0]['lr'], bot.step)
            bot.logger.tb_scalars(
                "loss", {"train": train_loss_avg}, bot.step)
            self.train_logs.append(train_loss_avg)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        self.metrics["step"].append(bot.step)
        history_length = len(self.metrics["step"])
        bot.logger.info(f"Metrics at step {bot.step}:")
        for metric_name, (metric_value, metric_string) in metrics.items():
            self.metrics[metric_name].append((metric_value, metric_string))
            assert history_length == len(
                self.metrics[metric_name]), "Inconsistent metric found!"
            bot.logger.info(f"{metric_name}: {metric_string}")
            bot.logger.tb_scalars(
                metric_name, {"val": metric_value}, bot.step)

    def on_train_ends(self, bot: BaseBot):
        if self.metrics["step"]:
            bot.logger.info("Training finished. Best step(s):")
            for metric_name, metric_values in self.metrics.items():
                if metric_name == "step":
                    continue
                best_idx = np.argmin(
                    np.array([x[0] for x in metric_values]))
                bot.logger.info(
                    "%s: %s @ step %d",
                    metric_name, metric_values[best_idx][1],
                    self.metrics["step"][best_idx]
                )

    def reset(self):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.metrics = defaultdict(list)
        self.timer = 0.0
        self.train_logs = []


class CheckpointCallback(Callback):
    """Save and manage checkpoints.

    You can resume training from a checkpoint.
    See `Basebot.load_checkpoint()`.
    """

    def __init__(
            self, keep_n_checkpoints: int = 1,
            checkpoint_dir: Path = Path("./data/cache/model_cache/"),
            monitor_metric: str = "loss"):
        super().__init__()
        assert keep_n_checkpoints > 0
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.monitor_metric = monitor_metric
        self.best_performers: List[Tuple[float, Path, int]] = []
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        target_value, target_string = metrics[self.monitor_metric]
        target_path = (
            self.checkpoint_dir /
            "ckpt_{}_{}_{}_{}.pth".format(
                bot.name, target_string, bot.step,
                datetime.now().strftime("%m%d%H%M"))
        )
        bot.logger.debug("Saving checkpoint %s...", target_path)
        if (
            len(self.best_performers) < self.keep_n_checkpoints or
            target_value < self.best_performers[-1][0]
        ):
            self.best_performers.append((target_value, target_path, bot.step))
            self.remove_checkpoints(keep=self.keep_n_checkpoints)
            torch.save(bot.state_dict(), target_path)
            assert Path(target_path).exists()

    def remove_checkpoints(self, keep):
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        for checkpoint in np.unique([
                x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def reset(self, ignore_previous=False):
        if ignore_previous:
            self.best_performers = []
        else:
            self.remove_checkpoints(0)


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int, min_improv: float, monitor_metric: str = "loss"):
        super().__init__()
        self.patience = patience
        self.min_improv = min_improv
        self.monitor_metric = monitor_metric
        self.reset()

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        target_value, _ = metrics[self.monitor_metric]
        if target_value < self.best - self.min_improv:
            bot.logger.info(
                "New low: %.6f improvement\n",
                self.best - target_value)
            self.best = target_value
            self.no_improv = 0
        else:
            self.no_improv += 1
        if self.no_improv > self.patience:
            raise StopTraining()

    def reset(self):
        self.no_improv = 0
        self.best = float('Inf')
