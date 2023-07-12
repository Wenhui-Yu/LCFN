## baseline: Spectral Collaborative Filtering (SCF)
## Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, and Philip S. Yu. Spectral collaborative filtering. In Proceedings of the 12th ACM Conference on Recommender Systems, RecSys '18, pages 311-319, 2018.

import tensorflow as tf
from utils.utils import *

def sampler_SCF(params, index):
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
    filters = []
    for l in range(layer):
        filters.append(tf.Variable(tf.random_normal([emb_dim, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_filters_' + str(l)))

    ## graph convolution
    embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)
    all_embeddings = [embeddings]
    for l in range(layer):
        ## convolution of embedding: (U*U^T+U*\Lambda*U^T)*emb = (I+L)*emb = (2*I-D^{-1}*A)*emb = 2*emb-H_hat*emb
        embeddings = 2 * embeddings - tf.sparse_tensor_dense_matmul(A_hat, embeddings)
        embeddings = tf.nn.sigmoid(tf.matmul(embeddings, filters[l]))
        all_embeddings.append(embeddings)
    all_embeddings = tf.concat(all_embeddings, 1)
    user_all_embeddings, item_all_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)

    ## lookup
    u_embeddings = tf.nn.embedding_lookup(user_all_embeddings, users)
    pos_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, pos_items)
    neg_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, neg_items)

    ## var collection
    var_set = [user_embeddings, item_embeddings] + filters
    reg_set = [u_embeddings, pos_i_embeddings, neg_i_embeddings]

    ## sampler scores
    samp_pos_scores = inner_product(u_embeddings, pos_i_embeddings)
    samp_neg_scores = inner_product(u_embeddings, neg_i_embeddings)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores