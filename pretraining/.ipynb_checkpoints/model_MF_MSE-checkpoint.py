## MF optimized by point-wise MSE
## all embeddings are restrained in the first octant by ReLU

import tensorflow as tf
import numpy as np

class model_MF_MSE(object):
    def __init__(self,n_users,n_items,emb_dim,lr,lamda):
        self.model_name = 'MF_MSE'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.user_embedding = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.002, dtype=tf.float32),
            name='user_embeddings')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.002, dtype=tf.float32),
            name='item_embeddings')

        self.user_embeddings = tf.nn.relu(self.user_embedding)
        self.item_embeddings = tf.nn.relu(self.item_embedding)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)

        #self.all_ratings = tf.matmul(self.user_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)

        self.loss = self.create_rmse_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings) + \
                    self.lamda * self.regularization(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embedding, self.item_embedding])

    def create_rmse_loss(self, users, pos_items, neg_items):
        if users.get_shape().as_list()[0] == None: length = 1
        else: length = users.get_shape().as_list()[0]
        ones = np.array([1] * length).astype(np.float32)
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        loss = tf.nn.l2_loss(ones-pos_scores)+tf.nn.l2_loss(neg_scores)
        return loss

    def regularization(self, users, pos_items, neg_items):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        return regularizer
