## author@Wenhui Yu  2021.02.16
## read train/test/validation data
## transform data into wanted structures
## return user and item number, and padding train data
## read (hyper-) graph embeddings, propoagation embeddings, and pre-trained embeddings
## construct sparse graph

import json
import numpy as np
import random as rd
from dense2sparse import propagation_matrix

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
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = 'dataset/' + DATASET + '/'
    train_path = DIR + 'train_data.json'
    test_path = DIR + 'test_data.json'
    validation_path = DIR + 'validation_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                     # hypergraph embeddings
    graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                         # 1d graph embeddings
    graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors
    if MODEL == 'SGNN': propagation_embeddings_path = DIR + 'pre_train_feature' + str(PROP_DIM) + '.json'   # pretrained latent factors

    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    ## load test data
    teat_vali_path = validation_path if TEST_VALIDATION == 'Validation' else test_path
    test_data = read_data(teat_vali_path)[0]
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False

    ## load pre-trained transform bases for LCFN and SGNN
    if MODEL == 'LCFN': hypergraph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    if MODEL == 'LGCN':
        if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY)
        if GRAPH_CONV == '2D_graph': graph_embeddings = read_bases(graph_embeddings_2d_path, FREQUENCY_USER, FREQUENCY_ITEM)
        if GRAPH_CONV == '2D_hyper_graph': graph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    if MODEL == 'SGNN':
        if PROP_EMB == 'RM': propagation_embeddings = read_bases(propagation_embeddings_path, PROP_DIM, PROP_DIM)
        if PROP_EMB == 'SF': propagation_embeddings = read_bases1(graph_embeddings_1d_path, PROP_DIM, IF_NORM)
        if PROP_EMB == 'PE': propagation_embeddings = 0

    ## convert dense graph to sparse graph
    if MODEL in ['GCMC', 'SCF', 'CGMC']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'left_norm')
    elif MODEL in ['NGCF', 'LightGCN']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')

    return train_data, train_data_interaction, user_num, item_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN