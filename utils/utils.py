import tensorflow as tf
from samplers.sampler_MF import *

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

def dlnrs_loss(scores, sampler, params, index):
    ## score of predicter
    pos_scores, neg_scores = scores

    ## score of sampler
    if sampler == "MF": samp_pos_scores, samp_neg_scores, var_set, reg_set = MF_sampler(params, index)
    if sampler == "NCF": samp_pos_scores, samp_neg_scores, var_set, reg_set = NCF_sampler(params, index)
    if sampler == "SCF": samp_pos_scores, samp_neg_scores, var_set, reg_set = SCF_sampler(params, index)
    if sampler == "LightGCN": samp_pos_scores, samp_neg_scores, var_set, reg_set = LightGCN_sampler(params, index)
    if sampler == "LGCN": samp_pos_scores, samp_neg_scores, var_set, reg_set = LGCN_sampler(params, index)
    pos_scores, neg_scores = tf.nn.sigmoid(pos_scores), tf.nn.sigmoid(neg_scores)
    samp_pos_scores, samp_neg_scores = tf.nn.sigmoid(samp_pos_scores), tf.nn.sigmoid(samp_neg_scores)

    pos_loss_predictor = tf.log(pos_scores)
    neg_loss_predictor = tf.log(1 - neg_scores) + tf.multiply(tf.stop_gradient(samp_neg_scores), tf.log(neg_scores))
    loss_predictor = tf.reduce_sum(pos_loss_predictor) + tf.reduce_sum(neg_loss_predictor)
    pos_loss_sampler = tf.multiply(tf.stop_gradient(pos_scores), tf.log(1 - samp_pos_scores))
    neg_loss_sampler = aux_loss_weight * tf.multiply(tf.stop_gradient(neg_scores), tf.log(samp_neg_scores)) + tf.log(1 - samp_neg_scores)
    loss_sampler = tf.reduce_sum(pos_loss_sampler) + tf.reduce_sum(neg_loss_sampler)
    loss = -(loss_predictor + loss_sampler)
    loss += params[-1] * regularization(reg_set)
    return loss, var_set

def MLP(emb, W, b):
    for l in range(len(W)):
        emb = tf.nn.relu(tf.matmul(emb, W[l]) + b[l])
    return emb

def regularization(reg_list):
    reg = 0
    for para in reg_list: reg += tf.nn.l2_loss(para)
    return reg