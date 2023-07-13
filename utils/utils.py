import tensorflow as tf
from samplers.sampler_MF import *
from samplers.sampler_NCF import *
from samplers.sampler_NGCF import *
from samplers.sampler_LightGCN import *
from samplers.sampler_LCFN import *

def inner_product(users, items):
    scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
    return scores
def bpr_loss(pos_scores, neg_scores):
    maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
    loss = tf.negative(tf.reduce_sum(maxi))
    return loss

def cross_entropy_loss(pos_scores, neg_scores):
    maxi = tf.log(tf.nn.sigmoid(pos_scores)) + tf.log(1 - tf.nn.sigmoid(neg_scores))
    loss = tf.negative(tf.reduce_sum(maxi))
    return loss

def mse_loss(pos_scores, neg_scores):
    loss = tf.nn.l2_loss(1 - pos_scores) + tf.nn.l2_loss(neg_scores)
    return loss

def wbpr_loss(pos_scores, neg_scores, popularity, item):
    popularity = tf.constant(popularity, name='popularity', dtype=tf.float32)
    weight = tf.nn.embedding_lookup(popularity, item)
    maxi = tf.log(weight) * tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
    loss = tf.negative(tf.reduce_sum(maxi))
    return loss

def dlnrs_loss(scores, params, params_sampler, index):
    ## score of predicter
    pos_scores, neg_scores = scores
    sampler, lamba, aux_loss_weight = params
    ## score of sampler
    if sampler == "MF": samp_pos_scores, samp_neg_scores, var_set, reg_set = sampler_MF(params_sampler, index)
    if sampler == "NCF": samp_pos_scores, samp_neg_scores, var_set, reg_set = sampler_NCF(params_sampler, index)
    if sampler == "NGCF": samp_pos_scores, samp_neg_scores, var_set, reg_set = sampler_NGCF(params_sampler, index)
    if sampler == "LightGCN": samp_pos_scores, samp_neg_scores, var_set, reg_set = sampler_LightGCN(params_sampler, index)
    if sampler == "LCFN": samp_pos_scores, samp_neg_scores, var_set, reg_set = sampler_LCFN(params_sampler, index)
    pos_scores, neg_scores = tf.nn.sigmoid(pos_scores), tf.nn.sigmoid(neg_scores)
    samp_pos_scores, samp_neg_scores = tf.nn.sigmoid(samp_pos_scores), tf.nn.sigmoid(samp_neg_scores)

    pos_loss_predictor = tf.log(pos_scores)
    neg_loss_predictor = tf.log(1 - neg_scores) + tf.multiply(tf.stop_gradient(samp_neg_scores), tf.log(neg_scores))
    loss_predictor = tf.reduce_sum(pos_loss_predictor) + tf.reduce_sum(neg_loss_predictor)
    pos_loss_sampler = tf.multiply(tf.stop_gradient(pos_scores), tf.log(1 - samp_pos_scores))
    neg_loss_sampler = (1 - aux_loss_weight) * tf.multiply(tf.stop_gradient(neg_scores), tf.log(samp_neg_scores)) + aux_loss_weight * tf.log(1 - samp_neg_scores)
    loss_sampler = tf.reduce_sum(pos_loss_sampler) + tf.reduce_sum(neg_loss_sampler)
    loss = -(loss_predictor + loss_sampler)
    loss += lamba * regularization(reg_set)
    return loss, var_set

def MLP(emb, W, b):
    for l in range(len(W)):
        emb = tf.nn.relu(tf.matmul(emb, W[l]) + b[l])
    return emb

def regularization(reg_list):
    reg = 0
    for para in reg_list: reg += tf.nn.l2_loss(para)
    return reg