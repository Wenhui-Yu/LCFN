
import tensorflow as tf
import numpy as np
from utils.utils import *

def sampler_LCFN(params, index):
    n_users, n_items, emb_dim, if_pretrain, _, graph_emb, U, V = params
    users, pos_items, neg_items = index
    P, Q = graph_emb
    frequence_user = int(np.shape(P)[1])
    frequence_item = int(np.shape(Q)[1])
    layer = 1
    layer_weight = [1 / (l + 1) for l in range(layer + 1)]

    ## trainable parameter
    if if_pretrain:
        user_embeddings = tf.Variable(U, name='samp_user_embeddings')
        item_embeddings = tf.Variable(V, name='samp_item_embeddings')
    else:
        user_embeddings = tf.Variable(tf.random_normal([n_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings')
        item_embeddings = tf.Variable(tf.random_normal([n_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings')
    user_filters = []
    item_filters = []
    transformers = []
    for l in range(layer):
        user_filters.append(tf.Variable(tf.random_normal([frequence_user], mean=1, stddev=0.001, dtype=tf.float32), name='user_filters_' + str(l)))
        item_filters.append(tf.Variable(tf.random_normal([frequence_item], mean=1, stddev=0.001, dtype=tf.float32), name='item_filters_' + str(l)))
        transformers.append(tf.Variable(tf.random.normal([emb_dim, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='transformers_' + str(l)))

    ## convolutional layers definition
    User_embedding = user_embeddings
    user_all_embeddings = [User_embedding]
    for l in range(layer):
        User_embedding = tf.matmul(tf.matmul(P, tf.diag(user_filters[l])), tf.matmul(P, User_embedding, transpose_a=True, transpose_b=False))
        User_embedding = tf.nn.sigmoid(tf.matmul(User_embedding, transformers[l]))
        user_all_embeddings.append(User_embedding)
    user_all_embeddings = tf.concat(user_all_embeddings, 1)

    Item_embedding = item_embeddings
    item_all_embeddings = [Item_embedding]
    for l in range(layer):
        Item_embedding = tf.matmul(tf.matmul(Q, tf.diag(item_filters[l])), tf.matmul(Q, Item_embedding, transpose_a=True, transpose_b=False))
        Item_embedding = tf.nn.sigmoid(tf.matmul(Item_embedding, transformers[l]))
        item_all_embeddings.append(Item_embedding)
    item_all_embeddings = tf.concat(item_all_embeddings, 1)

    ## lookup
    u_embeddings = tf.nn.embedding_lookup(user_all_embeddings, users)
    pos_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, pos_items)
    neg_i_embeddings = tf.nn.embedding_lookup(item_all_embeddings, neg_items)

    u_embeddings_reg = tf.nn.embedding_lookup(user_embeddings, users)
    pos_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, pos_items)
    neg_i_embeddings_reg = tf.nn.embedding_lookup(item_embeddings, neg_items)

    ## var collection
    var_set = [user_embeddings, item_embeddings] + user_filters + item_filters + transformers
    reg_set = [u_embeddings_reg, pos_i_embeddings_reg, neg_i_embeddings_reg]

    ## logits
    samp_pos_scores = inner_product(u_embeddings, pos_i_embeddings)
    samp_neg_scores = inner_product(u_embeddings, neg_i_embeddings)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores