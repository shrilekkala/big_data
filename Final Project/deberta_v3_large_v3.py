#比赛数据：https://www.kaggle.com/competitions/feedback-prize-effectiveness/data

#模型加载权重https://www.kaggle.com/datasets/darraghdog/feedback/cfg_ch_32c/fold-1/checkpoint_last_seed453209.pth

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import DataCollatorWithPadding
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, ContextPooler
from transformers import get_cosine_schedule_with_warmup

from sklearn.model_selection import  KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import string
import random
import os
import joblib
import gc
import copy
import time
import re
from tqdm import tqdm
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    seed = 2022
    max_length = 512
    epoch = 3
    train_batch_size = 8
    valid_batch_size = 16

    model_name = "microsoft/deberta-v3-large"
    token_name = "microsoft/deberta-v3-large"

    scheduler = "CosineAnnealingLR"
    learning_rate = 3e-6
    min_lr = 1e-6
    T_max = 500
    weight_decay = 0.005
    
    num_classes = 3
    n_fold = 5
    train_folding = [0,1,2,3,4]
    n_accumulate = 1
    freezing = False
    gradient_checkpoint = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOG_DIR = './log/'
    logger_name = 'deberta-V3-large-v3'
    output_dir = './output/'
    
INPUT_DIR = './feedback-prize-effectiveness-data/'
pretask_model_path = '/content/checkpoint_last_seed453209.pth'
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.token_name, use_fast=True)
CFG.tokenizer.model_max_length = CFG.max_length
# CFG.tokenizer.is_fast

def get_logger(filename=CFG.LOG_DIR+CFG.logger_name):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1

        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
#loss_tr = SmoothBCEwLogits(smoothing=0.001)
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

def criterion_ls(outputs, labels):
    return SmoothCrossEntropyLoss(smoothing=0.1)(outputs, labels)

def get_score(outputs, labels):
    outputs = F.softmax(torch.tensor(outputs)).numpy()
    return log_loss(labels, outputs)

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_freezed_parameters(module):
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters
           
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import Parameter
from torch.autograd.function import InplaceFunction


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p)
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None.
    # I modified the code of dropout in PyTorch.
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p))
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer,
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p)
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )


TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")

TEST_DIR = os.path.join(INPUT_DIR, "test")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")

df = pd.read_csv(TRAIN_CSV)#.head(200)
# df.columns

# Get Essay text for each rows
def get_essay(essay_id):
    path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    essay_text = open(path, 'r').read()
    return essay_text
    
df['essay_text'] = df['essay_id'].apply(get_essay)
#df.head()

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    text = re.sub(r' \Z', '', text)
    return text
df['discourse_text'] = df['discourse_text'].apply(lambda x : resolve_encodings_and_normalize(x))
df['essay_text'] = df['essay_text'].apply(lambda x : resolve_encodings_and_normalize(x))


df['discourse_effectiveness'] = df['discourse_effectiveness'].map({'Ineffective': 0, 'Adequate': 1, 'Effective': 2})
gkf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for fold, (train_id, val_id) in enumerate(gkf.split(X=df, y=df.discourse_effectiveness, groups=df.essay_id)):
    df.loc[val_id , "kfold"] = fold

#df = pd.concat([df, pdata], axis=0, ignore_index=True)

class FeedbackDataset(Dataset):
    def __init__(self,df, max_length, tokenizer, training=True):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse_type = self.df['discourse_type'].values
        self.discourse_text = self.df['discourse_text'].values
        self.essays = self.df['essay_text'].values
        self.training = training
        
        if self.training:
            self.targets = self.df['discourse_effectiveness'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        discourse_type = self.discourse_type[index]
        discourse_text = self.discourse_text[index]
        essay = self.essays[index]
        type_text = discourse_type + ' ' + discourse_text
        
        inputs = self.tokenizer.encode_plus(
            type_text, 
            essay,
            truncation = True,
            add_special_tokens = True,
            return_token_type_ids = True,
            max_length = self.max_len
        )
        
        samples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }
        
        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']
          
        if self.training:
            samples['target'] = self.targets[index]
        
        return samples
    
    
# Dynamic Padding (Collate)
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

# collate_fn = DataCollatorWithPadding(tokenizer=CFG.tokenizer)
collate_fn = Collate(CFG.tokenizer)
    
    
    
class FeedbackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedbackModel, self).__init__()
        
        # DeBERTa
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states":True, 
                      'return_dict':True}) 
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        # gradient checkpointing
        if CFG.gradient_checkpoint:
            self.backbone.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.backbone.is_gradient_checkpointing}")
        if  CFG.freezing:
            freeze(self.backbone.embeddings)
            freeze(self.backbone.encoder.layer[:6])
        
        #self.context_pooler = ContextPooler(self.config)
        #self.conv1 = nn.Conv1d(self.config.hidden_size, 512, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #self.conv2 = nn.Conv1d(512, 3, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        # Multi Sample Dropout
        self.fc = nn.Linear(self.config.hidden_size, CFG.num_classes)
        #self.multi_sample_dropout = MultiSampleDropout(self.fc, start_prob=0.1, num_samples=5, increment=0.1)
        #self._init_weights(self.conv1)
        #self._init_weights(self.conv2)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, ids, mask):        
        out = self.backbone(input_ids=ids,attention_mask=mask)
        hs = out['hidden_states']
        # out = self.weighted_pooler(out.hidden_states) # For WeightedLayerPooling
        # out = self.pooler(out, mask) # For MeanPooling
                
        #out = self.context_pooler(torch.stack(list(out.hidden_states), dim=0)) # For ContextPooler
        #out = self.context_pooler(out[0]) # For ContextPooler
        x = hs[-1][:, 0, :]#torch.cat([hs[-1], hs[-2]], -1)
        #print(x.size())
        #x = torch.mean(x, 0)
        #print(x.size())
        #conv1_logits = self.conv1(x.transpose(1, 2))
        #conv2_logits = self.conv2(conv1_logits)
        #logits = conv2_logits.transpose(1, 2)
        #x = torch.mean(logits, 1)
        # out = self.pooler(out.last_hidden_state, mask)

        #x = self.multi_sample_dropout(x)
        
        #out = self.pooler(out.last_hidden_state, mask)
        #out = self.bilstm(out)[0]
        #out = self.drop(out)
        #x = self.fc(x)
        logits1 = self.fc(self.dropout1(x))
        logits2 = self.fc(self.dropout2(x))
        logits3 = self.fc(self.dropout3(x))
        logits4 = self.fc(self.dropout4(x))
        logits5 = self.fc(self.dropout5(x))
        x = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return x

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    parameters = []
    for layer_num, (name, params) in enumerate(named_parameters):
        weight_decay = 0.0 if "bias" in name else 0.003

        lr = CFG.learning_rate

        if layer_num >= 129:        
            lr = 2*CFG.learning_rate

        if layer_num >= 250:
            lr = 3*CFG.learning_rate
        #if 'conv' in name:
        #    lr = 1e-4
        #    weight_decay = 0.0 if "bias" in name else 0.005
        parameters.append({"params": params,
                "weight_decay": weight_decay,
                "lr": lr})

    return AdamW(parameters)

def train_one_epoch(model, dataloader, device, epoch, optimizer, scheduler):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    dataset_size = 0
    running_loss= 0
    
    bar = tqdm(enumerate(dataloader), total= len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        
        batch_size = ids.size(0)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        loss = loss/CFG.n_accumulate
        #loss.backward()
        scaler.scale(loss).backward()
       
        if (step+1)% CFG.n_accumulate ==0:
            scaler.step(optimizer)
            scaler.update()
            
            #optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
                
        running_loss += (loss.item()*batch_size) * CFG.n_accumulate
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch = epoch, Train_loss = epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss
        
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss= 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        
        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        
        running_loss += (loss.item()*batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch = epoch, Valid_loss = epoch_loss)
    gc.collect()
    return epoch_loss

def prepare_loaders(fold):
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    
    train_dataset = FeedbackDataset(df_train, tokenizer=CFG.tokenizer, max_length=CFG.max_length)
    valid_dataset = FeedbackDataset(df_valid, tokenizer=CFG.tokenizer, max_length=CFG.max_length)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, collate_fn=collate_fn, 
                              num_workers=32, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, collate_fn=collate_fn,
                              num_workers=32, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

def apply_mixout(model, p):
    for sup_module in model.modules():
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(
                    module.in_features, module.out_features, bias, target_state_dict["weight"], p
                )
                new_module.load_state_dict(target_state_dict)
                setattr(sup_module, name, new_module)
    return model

def run_training(model, device, num_epochs, fold, train_loader, valid_loader):
    #wandb.watch(model, log_freq = 100)
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    optimizer = create_optimizer(model)                        
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=CFG.epoch * len(train_loader),
        num_warmup_steps=300)
    for epoch in range(1, num_epochs+1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, train_loader, device, epoch, optimizer, scheduler)
        valid_epoch_loss = valid_one_epoch(model, valid_loader, device, epoch)
        
        history['Train Loss'].append(train_epoch_loss)
        history['Eval Loss'].append(valid_epoch_loss)
        
        LOGGER.info({'Train Loss': train_epoch_loss})
        LOGGER.info({'Eval Loss': valid_epoch_loss})
        
        if valid_epoch_loss <= best_epoch_loss:
            LOGGER.info(f"Valid Loss Improved: {best_epoch_loss} -------> {valid_epoch_loss}")
            best_epoch_loss = valid_epoch_loss
            #run.summary['Best Loss']= best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            path = CFG.output_dir+f'{CFG.logger_name}_f{fold}.bin'
            torch.save(model.state_dict(), path)
            LOGGER.info('Model Saved')
        
    end = time.time()
    time_eclipsed = end-start
    LOGGER.info(f'Time complete in: {time_eclipsed//3600}h:{(time_eclipsed%3600)//60}m:{time_eclipsed%60}s')
    LOGGER.info(f'Best Loss: {best_epoch_loss}')
    
    model.load_state_dict(best_model_wts)
    
    return model, history
transformers.logging.set_verbosity_error()
state = torch.load(pretask_model_path, map_location=torch.device('cpu'))
del state['model']['backbone.embeddings.position_ids']
for fold in CFG.train_folding:
    LOGGER.info(f'================ Fold: {fold}/5folds =================')
    
    cfg = dict(CFG.__dict__)
    del cfg['__dict__'], cfg['__weakref__']

    
    train_loader, valid_loader = prepare_loaders(fold)
    model = FeedbackModel(CFG.model_name)
    LOGGER.info(model.load_state_dict(state['model'], strict=False))
    model.to(CFG.device)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    model, history = run_training(model, CFG.device, CFG.epoch, fold, train_loader, valid_loader)
    
    #run.finish()
    gc.collect() 
    #break
