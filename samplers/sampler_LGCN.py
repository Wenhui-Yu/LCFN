## our model: Low-pass Graph Convolutional Network (LGCN)
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import tensorflow as tf
import numpy as np
from utils.utils import *

def sampler_LGCN(params, index):
    n_users, n_items, emb_dim, if_pretrain, _, graph_emb, U, V = params
    users, pos_items, neg_items = index
    frequency = int(np.shape(graph_emb)[1])
    layer = 1
    layer_weight = [1 / (l + 1) for l in range(layer + 1)]

    ## trainable parameter
    if if_pretrain:
        user_embeddings = tf.Variable(U, name='samp_user_embeddings')
        item_embeddings = tf.Variable(V, name='samp_item_embeddings')
    else:
        user_embeddings = tf.Variable(tf.random_normal([n_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings')
        item_embeddings = tf.Variable(tf.random_normal([n_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings')
    kernel = [tf.Variable(tf.random_normal([frequency], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_filters_' + str(l)) for l in range(layer)]

    ## convolutional layers definition
    embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)
    all_embeddings = embeddings
    for l in range(layer):      ## low-pass graph convolution
        embeddings = tf.matmul(tf.matmul(graph_emb, tf.diag(kernel[l])), tf.matmul(graph_emb, embeddings, transpose_a=True, transpose_b=False))
        all_embeddings += embeddings * layer_weight[l + 1]
    user_all_embeddings, item_all_embeddings = tf.split(all_embeddings, [n_users, n_items], 0)

    ## lookup
    u_embeddings = tf.nn.embedding_lookup(user_all_embeddings, users)
    pos_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, pos_items)
    neg_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, neg_items)

    u_embeddings_reg = tf.nn.embedding_lookup(user_embeddings, users)
    pos_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, pos_items)
    neg_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, neg_items)

    ## var collection
    var_set = [user_embeddings, item_embeddings] + kernel
    reg_set = [u_embeddings_reg, pos_i_embeddings_reg, neg_i_embeddings_reg]

    ## logits
    samp_pos_scores = inner_product(u_embeddings, pos_i_embeddings)
    samp_neg_scores = inner_product(u_embeddings, neg_i_embeddings)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores