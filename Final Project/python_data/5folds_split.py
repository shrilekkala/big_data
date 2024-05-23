import pandas as pd 
import os 
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
import re
from sklearn.model_selection import  KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
class CFG:
    seed = 2022
    n_fold = 5
    num_classes = 3
    
INPUT_DIR = '../input/feedback-prize-effectiveness'
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")

TEST_DIR = os.path.join(INPUT_DIR, "train")
TEST_CSV = os.path.join(INPUT_DIR, "train.csv")

df = pd.read_csv(TRAIN_CSV)#.head(200)
def get_essay_old(essay_id):
    path = os.path.join('../input/feedback-prize-effectiveness/train', f"{essay_id}.txt")
    essay_text = open(path, 'r').read()
    return essay_text
    
df['essay_text'] = df['essay_id'].apply(get_essay_old)
df['discourse_effectiveness'] = df['discourse_effectiveness'].map({'Ineffective': 0, 'Adequate': 1, 'Effective': 2})
gkf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for fold, (train_id, val_id) in enumerate(gkf.split(X=df, y=df.discourse_effectiveness, groups=df.essay_id)):
    df.loc[val_id , "kfold"] = fold
df.to_csv('5fold_oof.csv', index=False)







