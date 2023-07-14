## baseline: Neural Graph Collaborative Filtering (NGCF)
## XiangWang, Xiangnan He, MengWang, Fuli Feng, and Tat-Seng Chua. 2019. Neural Graph Collaborative Filtering. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '19), 2019.

import tensorflow as tf
from utils.utils import *

def sampler_NGCF(params, index):
    n_users, n_items, emb_dim, if_pretrain, A_hat, _, U, V = params
    users, pos_items, neg_items = index
    layer = 1

    ## trainable parameter
    if if_pretrain:
        user_embeddings = tf.Variable(U, name='samp_user_embeddings')
        item_embeddings = tf.Variable(V, name='samp_item_embeddings')
    else:
        user_embeddings = tf.Variable(tf.random_normal([n_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings')
        item_embeddings = tf.Variable(tf.random_normal([n_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings')
    filters_1 = []
    filters_2 = []
    for l in range(layer):
        filters_1.append(tf.Variable(
            tf.random.normal([emb_dim, emb_dim], mean=0.0, stddev=0.001, dtype=tf.float32) + \
            tf.diag(tf.random.normal([emb_dim], mean=1.0, stddev=0.001, dtype=tf.float32)),
            name='samp_filter_1_' + str(l)
        ))
        filters_2.append(tf.Variable(
            tf.random.normal([emb_dim, emb_dim], mean=0.0, stddev=0.001, dtype=tf.float32) + \
            tf.diag(tf.random.normal([emb_dim], mean=1.0, stddev=0.001, dtype=tf.float32)),
            name='samp_filter_2_' + str(l)
        ))

    ## graph convolution
    embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)
    all_embeddings = [embeddings]
    for l in range(layer):
        propagations = tf.sparse_tensor_dense_matmul(A_hat, embeddings)
        embeddings_1 = propagations + embeddings
        embeddings_2 = tf.multiply(propagations, embeddings)
        embeddings = tf.nn.relu(tf.matmul(embeddings_1, filters_1[l]) + tf.matmul(embeddings_2, filters_2[l]))
        all_embeddings.append(embeddings)
    all_embeddings = tf.concat(all_embeddings, 1)
    user_all_embeddings, item_all_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)

    ## lookup
    u_embeddings = tf.nn.embedding_lookup(user_all_embeddings, users)
    pos_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, pos_items)
    neg_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, neg_items)

    u_embeddings_reg = tf.nn.embedding_lookup(user_embeddings, users)
    pos_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, pos_items)
    neg_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, neg_items)

    ## var collection
    var_set = [user_embeddings, item_embeddings] + filters_1 + filters_2
    reg_set = [u_embeddings_reg, pos_i_embeddings_reg, neg_i_embeddings_reg] + filters_1 + filters_2

    ## sampler scores
    samp_pos_scores = inner_product(u_embeddings, pos_i_embeddings)
    samp_neg_scores = inner_product(u_embeddings, neg_i_embeddings)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores