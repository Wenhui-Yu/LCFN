## baseline: Spectral Collaborative Filtering (SCF)
## Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, and Philip S. Yu. Spectral collaborative filtering. In Proceedings of the 12th ACM Conference on Recommender Systems, RecSys '18, pages 311-319, 2018.

import tensorflow as tf
from utils.utils import *

class model_SCF(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'SCF'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.layer = para['LAYER']
        self.if_pretrain = para['IF_PRETRAIN']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.popularity = data['popularity']
        self.U, self.V = data['pre_train_embeddings']
        self.A_hat = data['sparse_propagation_matrix']

        ## placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        ## define trainable parameters
        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
        else:
            self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
            self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
        self.filters = []
        for l in range(self.layer):
            self.filters.append(tf.Variable(tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='filters_' + str(l)))
        self.var_list = [self.user_embeddings, self.item_embeddings] + self.filters

        ## graph convolution
        self.embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        self.all_embeddings = [self.embeddings]
        for l in range(self.layer):
            ## convolution of embedding: (U*U^T+U*\Lambda*U^T)*emb = (I+L)*emb = (2*I-D^{-1}*A)*emb = 2*emb-H_hat*emb
            self.embeddings = 2 * self.embeddings - tf.sparse_tensor_dense_matmul(self.A_hat, self.embeddings)
            self.embeddings = tf.nn.sigmoid(tf.matmul(self.embeddings, self.filters[l]))
            self.all_embeddings.append(self.embeddings)
        self.all_embeddings = tf.concat(self.all_embeddings, 1)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(self.all_embeddings, [self.n_users, self.n_items], 0)

        ## lookup
        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        ## logits
        self.pos_scores = inner_product(self.u_embeddings, self.pos_i_embeddings)
        self.neg_scores = inner_product(self.u_embeddings, self.neg_i_embeddings)

        ## loss function
        if self.loss_function == 'BPR': self.loss = bpr_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'CrossEntropy': self.loss = cross_entropy_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'MSE': self.loss = mse_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'WBPR': self.loss = wbpr_loss(self.pos_scores, self.neg_scores, self.popularity)
        if self.loss_function == 'DLNRS':
            self.loss, self.samp_var = dlnrs_loss([self.pos_scores, self.neg_scores],
                                                  self.sampler,
                                                  [self.n_users, self.n_items, self.emb_dim, self.A_hat, self.lamda],
                                                  [self.users, self.pos_items, self.neg_items])
            self.var_list += self.samp_var

        ## regularization
        self.loss += self.lamda * regularization([self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings])

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
