## author@Wenhui Yu  2021.02.16
## read train/test/validation data
## transform data into wanted structures
## return user and item number, and padding train data
## read (hyper-) graph embeddings, propoagation embeddings, and pre-trained embeddings
## construct sparse graph

import json
import numpy as np
import random as rd
from utils.dense2sparse import propagation_matrix
from params.params_LGCN import FREQUENCY

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
    return data, interactions, user_num, item_num

def read_popularity(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    return data

def read_bases(path, fre_u, fre_v):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_v = np.array(feat_v)[:, 0: fre_v].astype(np.float32)
    return [feat_u, feat_v]

def read_bases1(path, fre, _if_norm = False):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    if _if_norm:
        for i in range(len(bases)):
            bases[i] = bases[i]/np.sqrt(np.dot(bases[i], bases[i]))
    return np.array(bases)[:, 0: fre].astype(np.float32)

def read_all_data(all_para):
    pre_train_embeddings = [0, 0]
    graph_embeddings, sparse_propagation_matrix = 0, 0
    ## Paths of data
    DIR = 'dataset/' + all_para['DATASET'] + '/'
    train_path = DIR + 'train_data.json'
    test_path = DIR + 'test_data.json'
    validation_path = DIR + 'validation_data.json'
    popularity_path = DIR + 'popularity.json'
    graph_embeddings_path = DIR + 'graph_embeddings.json'                         # graph embeddings
    pre_train_feature_path = DIR + 'pre_train_embeddings' + str(all_para['EMB_DIM']) + '.json'         # pretrained latent factors
    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    ## load test data
    teat_vali_path = validation_path if all_para['TEST_VALIDATION'] == 'Validation' else test_path
    test_data = read_data(teat_vali_path)[0]
    popularity = read_popularity(popularity_path)
    ## load pre-trained embeddings for all deep models
    if all_para['IF_PRETRAIN']:
        try: pre_train_embeddings = read_bases(pre_train_feature_path, all_para['EMB_DIM'], all_para['EMB_DIM'])
        except:
            print('ERROR!!! There is no pre-trained embeddings found')
            all_para['IF_PRETRAIN'] = False

    ## load pre-trained transform bases for LCFN and SGNN
    if all_para['SAMPLER'] == 'LGCN': graph_embeddings = read_bases1(graph_embeddings_path, FREQUENCY)
    if all_para['SAMPLER'] in ['NGCF', 'LightGCN']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
    if all_para['MODEL'] == 'LGCN': graph_embeddings = read_bases1(graph_embeddings_path, all_para['FREQUENCY'])
    if all_para['MODEL'] in ['NGCF', 'LightGCN']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
    return train_data, train_data_interaction, popularity, user_num, item_num, test_data, pre_train_embeddings, graph_embeddings, sparse_propagation_matrix
