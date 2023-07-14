

import tensorflow as tf
import numpy as np
from utils.utils import *

class model_LCFN(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'LCFN'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.layer = para['LAYER']
        self.if_pretrain = para['IF_PRETRAIN']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.popularity = data['popularity']
        self.U, self.V = data['pre_train_embeddings']
        self.graph_emb = data['graph_embeddings']
        [self.P, self.Q] = self.graph_emb
        self.frequence_user = int(np.shape(self.P)[1])
        self.frequence_item = int(np.shape(self.Q)[1])
        self.layer_weight = [1 / (l + 1) for l in range(self.layer + 1)]
        self.A_hat = data['sparse_propagation_matrix']

        ## placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        ## learnable parameters
        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
        else:
            self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
            self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
        self.user_filters = []
        self.item_filters = []
        self.transformers = []
        for l in range(self.layer):
            self.user_filters.append(tf.Variable(tf.random_normal([self.frequence_user], mean=1.0, stddev=0.001, dtype=tf.float32), name='user_filters_' + str(l)))
            self.item_filters.append(tf.Variable(tf.random_normal([self.frequence_item], mean=1.0, stddev=0.001, dtype=tf.float32), name='item_filters_' + str(l)))
            self.transformers.append(tf.Variable(
                tf.random.normal([self.emb_dim, self.emb_dim], mean=0.0, stddev=0.001, dtype=tf.float32) + \
                tf.diag(tf.random.normal([self.emb_dim], mean=1.0, stddev=0.001, dtype=tf.float32)),
                name='transformers_' + str(l)
            ))
        self.var_list = [self.user_embeddings, self.item_embeddings] + self.user_filters + self.item_filters + self.transformers ## learnable parameter list

        ## convolutional layers definition
        self.User_embedding = self.user_embeddings
        self.user_all_embeddings = [self.User_embedding]
        for l in range(self.layer):
            self.User_embedding = tf.matmul(tf.matmul(self.P, tf.diag(self.user_filters[l])), tf.matmul(self.P, self.User_embedding, transpose_a=True, transpose_b=False))
            self.User_embedding = tf.nn.sigmoid(tf.matmul(self.User_embedding, self.transformers[l]))
            self.user_all_embeddings.append(self.User_embedding)
        self.user_all_embeddings = tf.concat(self.user_all_embeddings, 1)

        self.Item_embedding = self.item_embeddings
        self.item_all_embeddings = [self.Item_embedding]
        for l in range(self.layer):
            self.Item_embedding = tf.matmul(tf.matmul(self.Q, tf.diag(self.item_filters[l])), tf.matmul(self.Q, self.Item_embedding, transpose_a=True, transpose_b=False))
            self.Item_embedding = tf.nn.sigmoid(tf.matmul(self.Item_embedding, self.transformers[l]))
            self.item_all_embeddings.append(self.Item_embedding)
        self.item_all_embeddings = tf.concat(self.item_all_embeddings, 1)

        ## lookup
        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        self.u_embeddings_reg = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)

        ## logits
        self.pos_scores = inner_product(self.u_embeddings, self.pos_i_embeddings)
        self.neg_scores = inner_product(self.u_embeddings, self.neg_i_embeddings)

        ## loss function
        if self.loss_function == 'BPR': self.loss = bpr_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'CrossEntropy': self.loss = cross_entropy_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'MSE': self.loss = mse_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'WBPR': self.loss = wbpr_loss(self.pos_scores, self.neg_scores, self.popularity, self.neg_items)
        if self.loss_function == 'DLNRS':
            self.loss, self.samp_var = dlnrs_loss([self.pos_scores, self.neg_scores],
                                                  [self.sampler, self.lamda, self.aux_loss_weight],
                                                  [self.n_users, self.n_items, self.emb_dim, self.if_pretrain, self.A_hat, self.graph_emb, self.U, self.V],
                                                  [self.users, self.pos_items, self.neg_items])
            self.var_list += self.samp_var

        ## regularization
        self.loss += self.lamda * regularization([self.u_embeddings_reg, self.pos_i_embeddings_reg, self.neg_i_embeddings_reg])

        ## optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

        ## get top k
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices
