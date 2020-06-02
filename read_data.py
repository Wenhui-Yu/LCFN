## read train/test/validation data and transformation bases
## transform data into wanted structures
## return user and item number, and padding train data
## author@Wenhui Yu  2020.06.02
## email: yuwh16@mails.tsinghua.edu.cn

import json
import numpy as np
import random as rd
import math

def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return(data, interactions, user_num, item_num)

def read_bases(path, fre_u, fre_v):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_v = np.array(feat_v)[:, 0: fre_v].astype(np.float32)
    return [feat_u, feat_v]
