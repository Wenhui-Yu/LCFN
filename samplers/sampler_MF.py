## basic baseline MF_BPR

import tensorflow as tf

def sampler_MF(params, index):
    n_users, n_items, emb_dim, _, _, _, _, _ = params
    users, pos_items, neg_items = index

    ## trainable parameter
    user_embeddings = tf.Variable(tf.random_normal([n_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_user_embeddings')
    item_embeddings = tf.Variable(tf.random_normal([n_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='samp_item_embeddings')

    ## lookup
    u_embeddings = tf.nn.embedding_lookup(user_embeddings, users)
    pos_i_embeddings = tf.nn.embedding_lookup(item_embeddings, pos_items)
    neg_i_embeddings = tf.nn.embedding_lookup(item_embeddings, neg_items)

    ## var collection
    var_set = [user_embeddings, item_embeddings]
    reg_set = [u_embeddings, pos_i_embeddings, neg_i_embeddings]

    ## sampler scores
    samp_pos_scores = inner_product(u_embeddings, pos_i_embeddings)
    samp_neg_scores = inner_product(u_embeddings, neg_i_embeddings)

    return samp_pos_scores, samp_neg_scores, var_set, reg_set

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores
