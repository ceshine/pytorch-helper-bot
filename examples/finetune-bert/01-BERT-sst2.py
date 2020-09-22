#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataclasses
from pathlib import Path

import nlp
import torch
import numpy as np
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from pytorch_helper_bot import (
    BaseBot, MovingAverageStatsTrackerCallback,  CheckpointCallback,
    LearningRateSchedulerCallback, MultiStageScheduler, Top1Accuracy,
    LinearLR
)


# In[2]:


CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True)


# Reference:
# 
#     * https://github.com/huggingface/nlp/blob/master/notebooks/Overview.ipynb

# In[3]:


dataset = nlp.load_dataset('glue', "sst2")


# In[4]:


set([x['label'] for x in dataset["train"]])


# In[5]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[6]:


# Tokenize our training dataset
def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer.batch_encode_plus(example_batch['sentence'], pad_to_max_length=True, max_length=64)
    return encodings


# In[7]:


# Format our dataset to outputs torch.Tensor to train a pytorch model
columns = ['input_ids', 'token_type_ids', 'attention_mask', "label"]
for subset in ("train", "validation"): 
    dataset[subset] = dataset[subset].map(convert_to_features, batched=True)
    dataset[subset].set_format(type='torch', columns=columns)


# In[8]:


tokenizer.decode(dataset['train'][6]["input_ids"].numpy())


# In[9]:


dataset['train'][0]["attention_mask"]


# In[10]:


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


# In[11]:


valid_idx, test_idx = train_test_split(list(range(len(dataset["validation"]))), test_size=0.5, random_state=42)


# In[12]:


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


# In[13]:


# Instantiate a PyTorch Dataloader around our dataset
train_loader = torch.utils.data.DataLoader(SST2Dataset(train_dict), batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(SST2Dataset(valid_dict), batch_size=32, drop_last=False)
test_loader = torch.utils.data.DataLoader(SST2Dataset(test_dict), batch_size=32, drop_last=False)


# In[14]:


@dataclasses.dataclass
class SST2Bot(BaseBot):
    log_dir = CACHE_DIR / "logs"
    
    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"

    @staticmethod
    def extract_prediction(output):
        return output[0]


# In[15]:


model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()


# In[16]:


# torch.nn.init.kaiming_normal_(model.classifier.weight)
# torch.nn.init.constant_(model.classifier.bias, 0)
# torch.nn.init.kaiming_normal_(model.bert.pooler.dense.weight)
# torch.nn.init.constant_(model.bert.pooler.dense.bias, 0);


# In[17]:


optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


# In[18]:


if APEX_AVAILABLE:
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1"
    )


# In[19]:


total_steps = len(train_loader) * 3

checkpoints = CheckpointCallback(
    keep_n_checkpoints=1,
    checkpoint_dir=CACHE_DIR / "model_cache/",
    monitor_metric="accuracy"
)
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
                CosineAnnealingLR(optimizer, lr_durations[1])
            ],
            start_at_epochs=break_points
        )
    ),
    checkpoints
]
    
bot = SST2Bot(
    model=model, 
    train_loader=train_loader,
    valid_loader=valid_loader, 
    clip_grad=10.,
    optimizer=optimizer, echo=True,
    criterion=torch.nn.CrossEntropyLoss(),
    callbacks=callbacks,
    pbar=False, use_tensorboard=False,
    use_amp=APEX_AVAILABLE,
    metrics=(Top1Accuracy(),)
)


# In[20]:


print(total_steps)
bot.train(
    total_steps=total_steps,
    checkpoint_interval=len(train_loader) // 2
)
bot.load_model(checkpoints.best_performers[0][1])
checkpoints.remove_checkpoints(keep=0)


# In[21]:


TARGET_DIR = CACHE_DIR / "sst2_bert_uncased"
TARGET_DIR.mkdir(exist_ok=True)
bot.model.save_pretrained(TARGET_DIR)


# In[22]:


bot.eval(valid_loader)


# In[23]:


bot.eval(test_loader)


# In[24]:


tokenizer.pad_token_id

