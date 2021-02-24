## proposed model, Low-pass Collaborative Filter Network (LCFN)
## Wenhui Yu and Zheng Qin. 2020. Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters. In ICML.
## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

import tensorflow as tf
from numpy import *
import numpy as np

class model_LCFN(object):
    def __init__(self, layer, n_users, n_items, emb_dim, graph_embeddings, lr, lamda, pre_train_latent_factor, if_pretrain):
        self.model_name = 'LCFN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        [self.P, self.Q] = graph_embeddings
        self.frequence_user = int(shape(self.P)[1])
        self.frequence_item = int(shape(self.Q)[1])
        self.layer = layer
        self.lamda = lamda
        self.lr = lr
        [self.U, self.V] = pre_train_latent_factor
        self.if_pretrain = if_pretrain

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
        else:
            self.user_embeddings = tf.Variable(
                tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings')
            self.item_embeddings = tf.Variable(
                tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings')

        self.user_filters = []
        for l in range(self.layer):
            self.user_filters.append(
                tf.Variable(
                    tf.random_normal([self.frequence_user], mean=1, stddev=0.001, dtype=tf.float32))
            )
        self.item_filters = []
        for l in range(self.layer):
            self.item_filters.append(
                tf.Variable(
                    tf.random_normal([self.frequence_item], mean=1, stddev=0.001, dtype=tf.float32))
            )

        self.transformers = []
        for l in range(self.layer):
            self.transformers.append(
                tf.Variable(
                    (np.random.normal(0, 0.001, (self.emb_dim, self.emb_dim)) + np.diag(np.random.normal(1, 0.001, self.emb_dim))).astype(np.float32)
                )
            )
        
        User_embedding = self.user_embeddings
        self.user_all_embeddings = [User_embedding]
        for l in range(self.layer):
            User_embedding = tf.matmul(tf.matmul(self.P, tf.diag(self.user_filters[l])),
                                       tf.matmul(self.P, User_embedding,
                                                 transpose_a=True, transpose_b=False))
            User_embedding = tf.nn.sigmoid(tf.matmul(User_embedding, self.transformers[l]))
            # User_embedding = tf.nn.relu(tf.matmul(User_embedding, self.transformers[l]))
            self.user_all_embeddings += [User_embedding]
        self.user_all_embeddings = tf.concat(self.user_all_embeddings, 1)
        Item_embedding = self.item_embeddings
        self.item_all_embeddings = [Item_embedding]
        for l in range(self.layer):
            Item_embedding = tf.matmul(tf.matmul(self.Q, tf.diag(self.item_filters[l])),
                                       tf.matmul(self.Q, Item_embedding,
                                                 transpose_a=True, transpose_b=False))
            Item_embedding = tf.nn.sigmoid(tf.matmul(Item_embedding, self.transformers[l]))
            #Item_embedding = tf.nn.relu(tf.matmul(Item_embedding, self.transformers[l]))
            self.item_all_embeddings += [Item_embedding]
        self.item_all_embeddings = tf.concat(self.item_all_embeddings, 1)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        self.u_embeddings_reg = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)

        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings) + \
                    self.lamda*self.regularization(self.u_embeddings_reg, self.pos_i_embeddings_reg, self.neg_i_embeddings_reg,
                                                   self.user_filters, self.item_filters, self.transformers)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings] 
                                                             + self.user_filters + self.item_filters + self.transformers)

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def regularization(self, users, pos_items, neg_items, filter_u, filter_v, transform):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        for l in range(self.layer):
            regularizer += tf.nn.l2_loss(filter_u[l])+tf.nn.l2_loss(filter_v[l])+tf.nn.l2_loss(transform[l])
        return regularizer

    
