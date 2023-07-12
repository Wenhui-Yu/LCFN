## baseline: Neural Collaborative Filtering (NCF)
## Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web, WWW '17, pages 173-182, 2017.

import tensorflow as tf
from utils.utils import *

def sampler_NCF(params, index):
    n_users, n_items, emb_dim, if_pretrain, _, _, U, V = params
    users, pos_items, neg_items = index
    layer = 1
    weight_size_list = [emb_dim]
    for l in range(layer):
        weight_size_list.append(max(int(0.5 ** l * 64), 4))

    ## trainable parameter
    if if_pretrain:
        user_embeddings_GMF = tf.Variable(U[:, : int(emb_dim / 2)], name='samp_user_embeddings_GMF')
        item_embeddings_GMF = tf.Variable(V[:, : int(emb_dim / 2)], name='samp_item_embeddings_GMF')
        user_embeddings_MLP = tf.Variable(U[:, int(emb_dim / 2):], name='samp_user_embeddings_MLP')
        item_embeddings_MLP = tf.Variable(V[:, int(emb_dim / 2):], name='samp_item_embeddings_MLP')
    else:
        user_embeddings_GMF = tf.Variable(tf.random_normal([n_users, int(emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings_GMF')
        item_embeddings_GMF = tf.Variable(tf.random_normal([n_items, int(emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings_GMF')
        user_embeddings_MLP = tf.Variable(tf.random_normal([n_users, int(emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings_MLP')
        item_embeddings_MLP = tf.Variable(tf.random_normal([n_items, int(emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings_MLP')
    W = []
    b = []
    for l in range(layer):
        W.append(tf.Variable(tf.random_normal([weight_size_list[l], weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
        b.append(tf.Variable(tf.random_normal([1, weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
    h = tf.Variable(tf.random_normal([1, int(emb_dim/2) + weight_size_list[-1]], mean=0.01, stddev=0.02, dtype=tf.float32), name='h')

    ## lookup
    u_embeddings_GMF = tf.nn.embedding_lookup(user_embeddings_GMF, users)
    pos_i_embeddings_GMF = tf.nn.embedding_lookup(item_embeddings_GMF, pos_items)
    neg_i_embeddings_GMF = tf.nn.embedding_lookup(item_embeddings_GMF, neg_items)
    u_embeddings_MLP = tf.nn.embedding_lookup(user_embeddings_MLP, users)
    pos_i_embeddings_MLP = tf.nn.embedding_lookup(item_embeddings_MLP, pos_items)
    neg_i_embeddings_MLP = tf.nn.embedding_lookup(item_embeddings_MLP, neg_items)

    ## var collection
    var_set = [user_embeddings_GMF, item_embeddings_GMF, user_embeddings_MLP, item_embeddings_MLP, h] + W + b
    reg_set = [u_embeddings_GMF, pos_i_embeddings_GMF, neg_i_embeddings_GMF,
               u_embeddings_MLP, pos_i_embeddings_MLP, neg_i_embeddings_MLP]

    ## logits
    samp_pos_scores = predict(u_embeddings_GMF, pos_i_embeddings_GMF, u_embeddings_MLP, pos_i_embeddings_MLP, W, b, h)
    samp_neg_scores = predict(u_embeddings_GMF, neg_i_embeddings_GMF, u_embeddings_MLP, neg_i_embeddings_MLP, W, b, h)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def predict(user_GMF, item_GMF, user_MLP, item_MLP, W, b, h):
    emb_GMF = tf.multiply(user_GMF, item_GMF)
    emb_MLP = MLP(tf.concat([user_MLP, item_MLP], axis=1), W, b)
    emb = tf.concat([emb_GMF, emb_MLP], axis=1)
    return tf.reshape(tf.matmul(emb, h, transpose_a=False, transpose_b=True), [-1])  # reshpae is not necessary with bpr loss but crutial with cross entropy loss

def MLP(emb, W, b):
    for l in range(len(W)):
        emb = tf.nn.relu(tf.matmul(emb, W[l]) + b[l])
    return emb
