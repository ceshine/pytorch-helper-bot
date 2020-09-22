""" Finetuning BERT using DeepSpeed's ZeRO-Offload
"""
import json
import dataclasses
from pathlib import Path
from functools import partial

import nlp
import torch
import typer
import deepspeed
import numpy as np
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split

from pytorch_helper_bot import (
    DeepSpeedBot, MovingAverageStatsTrackerCallback,  CheckpointCallback,
    LearningRateSchedulerCallback, MultiStageScheduler, Top1Accuracy,
    LinearLR, CosineAnnealingScheduler
)

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True)
APP = typer.Typer()


class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, entries_dict):
        super().__init__()
        self.entries_dict = entries_dict

    def __len__(self):
        return len(self.entries_dict["label"])

    def __getitem__(self, idx):
        return (
            self.entries_dict["input_ids"][idx],
            self.entries_dict["attention_mask"][idx],
            self.entries_dict["token_type_ids"][idx],
            self.entries_dict["label"][idx]
        )


@dataclasses.dataclass
class SST2Bot(DeepSpeedBot):
    log_dir = CACHE_DIR / "logs"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"

    @staticmethod
    def extract_prediction(output):
        return output[0]


class Object(object):
    pass


def convert_to_features(tokenizer, example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer.batch_encode_plus(
        example_batch['sentence'], padding='max_length', max_length=64, truncation=True)
    return encodings


@APP.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(arch="bert-base-uncased", config="gpu.json"):
    # Reference:
    #
    #     * https://github.com/huggingface/nlp/blob/master/notebooks/Overview.ipynb
    with open(config) as fin:
        config_params = json.load(fin)

    dataset = nlp.load_dataset('glue', "sst2")
    print(set([x['label'] for x in dataset["train"]]))

    tokenizer = BertTokenizerFast.from_pretrained(arch)

    # Format our dataset to outputs torch.Tensor to train a pytorch model
    columns = ['input_ids', 'token_type_ids', 'attention_mask', "label"]
    for subset in ("train", "validation"):
        dataset[subset] = dataset[subset].map(
            partial(convert_to_features, tokenizer), batched=True)
        dataset[subset].set_format(type='torch', columns=columns)

    print(tokenizer.decode(dataset['train'][6]["input_ids"].numpy()))
    print(dataset['train'][0]["attention_mask"])

    valid_idx, test_idx = train_test_split(
        list(range(len(dataset["validation"]))), test_size=0.5, random_state=42)

    train_dict = {
        "input_ids": dataset['train']["input_ids"],
        "attention_mask": dataset['train']["attention_mask"],
        "token_type_ids": dataset['train']["token_type_ids"],
        "label": dataset['train']["label"]
    }
    valid_dict = {
        "input_ids": dataset['validation']["input_ids"][valid_idx],
        "attention_mask": dataset['validation']["attention_mask"][valid_idx],
        "token_type_ids": dataset['validation']["token_type_ids"][valid_idx],
        "label": dataset['validation']["label"][valid_idx]
    }
    test_dict = {
        "input_ids": dataset['validation']["input_ids"][test_idx],
        "attention_mask": dataset['validation']["attention_mask"][test_idx],
        "token_type_ids": dataset['validation']["token_type_ids"][test_idx],
        "label": dataset['validation']["label"][test_idx]
    }

    # Instantiate a PyTorch Dataloader around our dataset
    train_loader = torch.utils.data.DataLoader(
        SST2Dataset(train_dict), batch_size=config_params["train_batch_size"], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        SST2Dataset(valid_dict), batch_size=config_params["train_batch_size"], drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        SST2Dataset(test_dict), batch_size=config_params["train_batch_size"], drop_last=False)

    model = BertForSequenceClassification.from_pretrained(arch)
    # torch.nn.init.kaiming_normal_(model.classifier.weight)
    # torch.nn.init.constant_(model.classifier.bias, 0)
    # torch.nn.init.kaiming_normal_(model.bert.pooler.dense.weight)
    # torch.nn.init.constant_(model.bert.pooler.dense.bias, 0);

    args = Object()
    setattr(args, "local_rank", 0)
    setattr(args, "deepspeed_config", config)
    if config[:3] == "cpu":
        if "optimizer" in config_params:
            model, optimizer, _, _ = deepspeed.initialize(
                args=args,
                model=model,
                model_parameters=model.parameters()
            )
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(model.parameters(), lr=2e-5)
            model, optimizer, _, _ = deepspeed.initialize(
                args=args,
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer
            )
    else:
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters()
            # optimizer=optimizer
        )

    total_steps = len(train_loader) * 3

    # checkpoints = CheckpointCallback(
    #     keep_n_checkpoints=1,
    #     checkpoint_dir=CACHE_DIR / "model_cache/",
    #     monitor_metric="accuracy"
    # )
    lr_durations = [
        int(total_steps*0.2),
        int(np.ceil(total_steps*0.8))
    ]
    break_points = [0] + list(np.cumsum(lr_durations))[:-1]
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_loader) // 8,
            log_interval=len(train_loader) // 10
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingScheduler(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            )
        ),
        # checkpoints
    ]

    bot = SST2Bot(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=torch.nn.CrossEntropyLoss(),
        callbacks=callbacks,
        pbar=False,
        use_tensorboard=False,
        # use_amp=APEX_AVAILABLE,
        metrics=(Top1Accuracy(),)
    )

    print(total_steps)
    bot.train(
        total_steps=total_steps,
        checkpoint_interval=len(train_loader) // 2
    )
    # bot.load_model(checkpoints.best_performers[0][1])
    # checkpoints.remove_checkpoints(keep=0)

    # TARGET_DIR = CACHE_DIR / "sst2_bert_uncased"
    # TARGET_DIR.mkdir(exist_ok=True)
    # bot.model.save_pretrained(TARGET_DIR)
    bot.eval(valid_loader)

    bot.eval(test_loader)


if __name__ == "__main__":
    APP()
