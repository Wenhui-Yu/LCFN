## author @Wenhui Yu  2021.01.24
## split train data into batches and train the model

from model_MF import model_MF
from model_NCF import model_NCF
from model_GCMC import model_GCMC
from model_NGCF import model_NGCF
from model_SCF import model_SCF
from model_CGMC import model_CGMC
from model_LightGCN import model_LightGCN
from model_LCFN import model_LCFN
from model_LGCN import model_LGCN
from model_SGNN import model_SGNN
from test_model import test_model
from print_save import print_value, save_value
import tensorflow as tf
import numpy as np
import random as rd
import pandas as pd
import time

def train_model(para, data, path_excel):
    ## data and hyperparameters
    [train_data, train_data_interaction, user_num, item_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, _] = data
    [_, _, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN, _, TOP_K] = para[0:13]
    if MODEL == 'LGCN': [_, _, _, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING] = para[13:]
    if MODEL == 'SGNN': [_, PROP_EMB, _] = para[13:]
    para_test = [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH]
    ## Define the model
    if MODEL == 'MF': model = model_MF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA)
    if MODEL == 'NCF': model = model_NCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'GCMC': model = model_GCMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'NGCF': model = model_NGCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'SCF': model = model_SCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'CGMC': model = model_CGMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LightGCN': model = model_LightGCN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LCFN': model = model_LCFN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, graph_embeddings=hypergraph_embeddings)
    if MODEL == 'LGCN': model = model_LGCN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    if MODEL == 'SGNN': model = model_SGNN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, propagation_embeddings=propagation_embeddings, if_pretrain=IF_PRETRAIN, prop_emb=PROP_EMB)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    t1 = time.clock()
    for epoch in range(N_EPOCH):
        for batch_num in range(len(batches) - 1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num + 1]):
                (user, pos_item) = train_data_interaction[sample]
                sample_num = 0
                while sample_num < (SAMPLE_RATE if MODEL == 'LGCN' else 1):
                    neg_item = int(rd.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, pos_item, neg_item])
            train_batch_data = np.array(train_batch_data)
            _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2], model.keep_prob: KEEP_PORB if MODEL == 'LGCN' else 1})
        ## test the model each epoch
        F1, NDCG = test_model(sess, model, para_test)
        F1_max = max(F1_max, F1[0])
        ## print performance
        # print_value([epoch + 1, loss, F1_max, F1, NDCG])
        if epoch % 10 == 0: print('%.5f' % (F1_max), end = ' ', flush = True)
        ## save performance
        F1_df.loc[epoch + 1] = F1
        NDCG_df.loc[epoch + 1] = NDCG
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if loss > 10 ** 10: break
    t2 = time.clock()
    print('time cost:', (t2 - t1) / 200)
    return F1_max