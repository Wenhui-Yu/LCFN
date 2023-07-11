import tensorflow as tf

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

def MLP(emb, W, b):
    for l in range(len(W)):
        emb = tf.nn.relu(tf.matmul(emb, W[l]) + b[l])
    return emb

def regularization(reg_list):
    reg = 0
    for para in reg_list: reg += tf.nn.l2_loss(para)
    return reg