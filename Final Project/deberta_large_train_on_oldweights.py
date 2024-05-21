#比赛数据：https://www.kaggle.com/competitions/feedback-prize-effectiveness/data
##模型加载权重：https://www.kaggle.com/datasets/cdeotte/deberta-large-100
cfg = {
    "num_proc": 8,
    # data
    "k_folds": 5,
    "max_length": 2048,
    "padding": False,
    "stride": 0,
    "data_dir": "./",
    "load_from_disk": None, # if you already tokenized, you can load it through this
    "pad_multiple": 8,
    # model
    "model_name_or_path": "microsoft/deberta-large",
    "dropout": 0.1,
    # to put in TrainingArguments
    "trainingargs": {
        "output_dir": "output",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "optim": 'adamw_torch',
        "logging_steps": 100,
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps":100,
        "report_to": "none",
        "group_by_length": False,
        "save_total_limit": 1,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "seed": 18,
        "fp16": True,
        "label_smoothing_factor":0.00,
        "gradient_checkpointing":True,
        "max_grad_norm":1.0,
        # you should probably set "fp16" to True, but it doesn't really matter on Kaggle
    }
}
import re
import pickle
import codecs
import warnings
import logging
from functools import partial
from pathlib import Path
from itertools import chain
from text_unidecode import unidecode
from typing import Any, Optional, Tuple
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, set_seed
import torch.utils.checkpoint

from datasets import Dataset, load_from_disk
        
import gc
import torch
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification

from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import (AutoTokenizer, 
                          DataCollatorForTokenClassification, 
                          TrainingArguments, 
                          Trainer,
                          TrainerCallback)
from torch import nn
import torch
oof = pd.read_csv('/content/5fold_oof.csv')

# https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330
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
    return text


def read_text_files(example, data_dir):

    id_ = example["essay_id"]

    with open(data_dir / "train" / f"{id_}.txt", "r") as fp:
        example["text"] = resolve_encodings_and_normalize(fp.read())

    return example

set_seed(cfg["trainingargs"]["seed"])

# change logging to not be bombarded by messages
# if you are debugging, the messages will likely be helpful
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)
data_dir = Path(cfg["data_dir"])

if cfg["load_from_disk"]:
    if not cfg["load_from_disk"].endswith(".dataset"):
        cfg["load_from_disk"] += ".dataset"
    ds = load_from_disk(cfg['load_from_disk'])
    
    pkl_file = f"{cfg['load_from_disk'][:-len('.dataset')]}_pkl"
    with open(pkl_file, "rb") as fp:
        grouped = pickle.load(fp)
    
    print("Loading from saved files")
else:
    train_df = pd.read_csv(data_dir / "train.csv")
    #train_df = pd.concat([train_df, pdata], ignore_index=True)
    text_ds = Dataset.from_dict({"essay_id": train_df.essay_id.unique()})

    text_ds = text_ds.map(
        partial(read_text_files, data_dir=data_dir),
        num_proc=cfg["num_proc"],
        batched=False,
        desc="Loading text files",
    )

    text_df = text_ds.to_pandas()

    train_df["discourse_text"] = [
        resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]
    ]

    train_df = train_df.merge(text_df, on="essay_id", how="left")

disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]
cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

label2id = {
    "Adequate": 0,
    "Effective": 1,
    "Ineffective": 2,
}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
tokenizer.add_special_tokens(
    {"additional_special_tokens": list(cls_tokens_map.values())+list(end_tokens_map.values())}
)
cls_id_map = {
    label: tokenizer.encode(tkn)[1]
    for label, tkn in cls_tokens_map.items()
}

def find_positions(example):

    text = example["text"][0]
    
    # keeps track of what has already
    # been located
    min_idx = 0
    
    # stores start and end indexes of discourse_texts
    idxs = []
    
    for dt in example["discourse_text"]:
        # calling strip is essential
        matches = list(re.finditer(re.escape(dt.strip()), text))
        
        # If there are multiple matches, take the first one
        # that is past the previous discourse texts.
        if len(matches) > 1:
            for m in matches:
                if m.start() >= min_idx:
                    break
        # If no matches are found
        elif len(matches) == 0:
            idxs.append([-1]) # will filter out later
            continue  
        # If one match is found
        else:
            m = matches[0]
            
        idxs.append([m.start(), m.end()])

        min_idx = m.start()

    return idxs

def tokenize(example):
    example["idxs"] = find_positions(example)

    text = example["text"][0]
    chunks = []
    labels = []
    prev = 0

    zipped = zip(
        example["idxs"],
        example["discourse_type"],
        example["discourse_effectiveness"],
    )
    for idxs, disc_type, disc_effect in zipped:
        # when the discourse_text wasn't found
        if idxs == [-1]:
            continue

        s, e = idxs

        # if the start of the current discourse_text is not 
        # at the end of the previous one.
        # (text in between discourse_texts)
        if s != prev:
            chunks.append(text[prev:s])
            prev = s

        # if the start of the current discourse_text is 
        # the same as the end of the previous discourse_text
        if s == prev:
            chunks.append(cls_tokens_map[disc_type])
            chunks.append(text[s:e])
            chunks.append(end_tokens_map[disc_type])
        
        prev = e

        labels.append(label2id[disc_effect])

    tokenized = tokenizer(
        " ".join(chunks),
        padding=False,
        truncation=True,
        max_length=cfg["max_length"],
        add_special_tokens=True,
    )
    
    # at this point, labels is not the same shape as input_ids.
    # The following loop will add -100 so that the loss function
    # ignores all tokens except CLS tokens

    # idx for labels list
    idx = 0
    final_labels = []
    for id_ in tokenized["input_ids"]:
        # if this id belongs to a CLS token
        if id_ in cls_id_map.values():
            final_labels.append(labels[idx])
            idx += 1
        else:
            # -100 will be ignored by loss function
            final_labels.append(-100)
    
    tokenized["labels"] = final_labels

    return tokenized

# I frequently restart my notebook, so to reduce time
# you can set this to just load the tokenized dataset from disk.
# It gets loaded in the 3rd code cell, but a check is done here
# to skip tokenizing
if cfg["load_from_disk"] is None:

    # make lists of discourse_text, discourse_effectiveness
    # for each essay
    grouped = train_df.groupby(["essay_id"]).agg(list)

    ds = Dataset.from_pandas(grouped)

    ds = ds.map(
        tokenize,
        batched=False,
        num_proc=cfg["num_proc"],
        desc="Tokenizing",
    )

    save_dir = f"{cfg['trainingargs']['output_dir']}"
    ds.save_to_disk(f"{save_dir}.dataset")
    with open(f"{save_dir}_pkl", "wb") as fp:
        pickle.dump(grouped, fp)
    print("Saving dataset to disk:", cfg['trainingargs']['output_dir'])
    


# basic kfold 
def get_folds(df, k_folds=5):

    kf = KFold(n_splits=k_folds)
    return [
        val_idx
        for _, val_idx in kf.split(df)
    ]

#fold_idxs = get_folds(ds["labels"], cfg["k_folds"])

origin = grouped.reset_index()

import numpy as np
fold_idxs = [np.asarray(origin[origin['essay_id'].isin(oof[oof['kfold']==0]['essay_id'].unique())].index.tolist()), 
            np.asarray(origin[origin['essay_id'].isin(oof[oof['kfold']==1]['essay_id'].unique())].index.tolist()), 
            np.asarray(origin[origin['essay_id'].isin(oof[oof['kfold']==2]['essay_id'].unique())].index.tolist()), 
            np.asarray(origin[origin['essay_id'].isin(oof[oof['kfold']==3]['essay_id'].unique())].index.tolist()), 
            np.asarray(origin[origin['essay_id'].isin(oof[oof['kfold']==4]['essay_id'].unique())].index.tolist()), 
            #np.asarray(origin[~origin['essay_id'].isin(oof['essay_id'].unique())].index.tolist()),
             ]
bad_matches = []
cls_ids = set(list(cls_id_map.values()))
for id_, l, ids, dt in zip(ds["essay_id"], ds["labels"], ds["input_ids"], grouped.discourse_text):
    
    # count number of labels (ignoring -100)
    num_cls_label = sum([x!=-100 for x in l])
    # count number of cls ids
    num_cls_id = sum([x in cls_ids for x in ids])
    # true number of discourse_texts
    num_dt = len(dt)
    
    if num_cls_label != num_dt or num_cls_id != num_dt:
        bad_matches.append((id_, l, ids, dt))
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerTokenClassifierOutput, LongformerModel
from torch import nn
import torch

class LongformerForTokenClassificationwithbiLSTM(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bigru = nn.GRU(config.hidden_size, (config.hidden_size) // 2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bigru(sequence_output)
        logits = self.classifier(lstm_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

args = TrainingArguments(**cfg["trainingargs"])
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.bestScore = np.inf

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.evaluation_strategy != "no", "SaveBestModelCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get("eval_loss")
        if metric_value < self.bestScore:
            print(f"** logloss score improved from {np.round(self.bestScore, 4)} to {np.round(metric_value, 4)} **")
            self.bestScore = metric_value
            control.should_save = True
        else:
            print(f"logloss score {np.round(metric_value, 4)} (Prev. Best {np.round(self.bestScore, 4)}) ")
# If using longformer, you will want to pad to a multiple of 512
# For most others, you'll want to pad to a multiple of 8
collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, pad_to_multiple_of=cfg["pad_multiple"], padding=True
)

output = args.output_dir
for fold in range(cfg["k_folds"]):
    
    args.output_dir = f"{output}-fold{fold}"
    
    model_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
        )
    model_config.update(
        {
            "num_labels": 3,
            "cls_tokens": list(cls_id_map.values()),
            "label2id": label2id,
            "id2label": {v:k for k, v in label2id.items()},
        }
    )
    
    model = AutoModelForTokenClassification.from_pretrained(cfg["model_name_or_path"], config=model_config)
    state = torch.load(f'/content/deberta-large-100/fold{fold}/pytorch_model.bin')
    del state['classifier.weight'], state['classifier.bias']
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(len(tokenizer)) 

    # split dataset to train and eval
    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_idxs =  list(chain(*[i for f, i in enumerate(fold_idxs) if f != fold]))
    train_dataset = ds.select(train_idxs).remove_columns([c for c in ds.column_names if c not in keep_cols])
    eval_dataset = ds.select(fold_idxs[fold]).remove_columns([c for c in ds.column_names if c not in keep_cols])


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        #compute_metrics=TextOverlapFBetaScore(test_df=valid_df, test_dataset=valid_dataset),
        callbacks=[SaveBestModelCallback],
    )
    
    trainer.train()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    #break

import json
best_metrics = []

for fold in range(cfg["k_folds"]):
    folder = Path(f"{output}-fold{fold}")
    checkpoint = sorted(list(folder.glob("checkpoint*")))[0]
    with open(checkpoint/"trainer_state.json", "r") as fp:
        data = json.load(fp)
        best_metrics.append(data["best_metric"])
    
print(best_metrics)
average = sum(best_metrics)/len(best_metrics)
average
