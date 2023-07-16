## baseline: Neural Collaborative Filtering (NCF)
## Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web, WWW '17, pages 173-182, 2017.

import tensorflow as tf
from utils.utils import *

class model_NCF(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'NCF'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.layer = para['LAYER']
        self.if_pretrain = para['IF_PRETRAIN']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        self.rho = para['RHO']
        self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.popularity = data['popularity']
        [self.U, self.V] = data['pre_train_embeddings']
        self.weight_size_list = [self.emb_dim]
        for l in range(self.layer):
            self.weight_size_list.append(max(int(0.5 ** l * 64), 4))
        self.A_hat = data['sparse_propagation_matrix']
        self.graph_emb = data['graph_embeddings']

        ## placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        ## define trainable parameters
        if self.if_pretrain:
            self.user_embeddings_GMF = tf.Variable(self.U[:, : int(self.emb_dim/2)], name='user_embeddings_GMF')
            self.item_embeddings_GMF = tf.Variable(self.V[:, : int(self.emb_dim/2)], name='item_embeddings_GMF')
            self.user_embeddings_MLP = tf.Variable(self.U[:, int(self.emb_dim/2):], name='user_embeddings_MLP')
            self.item_embeddings_MLP = tf.Variable(self.V[:, int(self.emb_dim/2):], name='item_embeddings_MLP')
        else:
            self.user_embeddings_GMF = tf.Variable(tf.random_normal([self.n_users, int(self.emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings_GMF')
            self.item_embeddings_GMF = tf.Variable(tf.random_normal([self.n_items, int(self.emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_GMF')
            self.user_embeddings_MLP = tf.Variable(tf.random_normal([self.n_users, int(self.emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings_MLP')
            self.item_embeddings_MLP = tf.Variable(tf.random_normal([self.n_items, int(self.emb_dim/2)], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_MLP')
        self.W = []
        self.b = []
        for l in range(self.layer):
            self.W.append(tf.Variable(tf.random_normal([self.weight_size_list[l], self.weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
            self.b.append(tf.Variable(tf.random_normal([1, self.weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
        self.h = tf.Variable(tf.random_normal([1, int(self.emb_dim/2) + self.weight_size_list[-1]], mean=0.01, stddev=0.02, dtype=tf.float32), name='h')
        self.var_list = [self.user_embeddings_GMF, self.item_embeddings_GMF, self.user_embeddings_MLP, self.item_embeddings_MLP, self.h] + self.W + self.b

        ## lookup
        self.u_embeddings_GMF = tf.nn.embedding_lookup(self.user_embeddings_GMF, self.users)
        self.pos_i_embeddings_GMF = tf.nn.embedding_lookup(self.item_embeddings_GMF, self.pos_items)
        self.neg_i_embeddings_GMF = tf.nn.embedding_lookup(self.item_embeddings_GMF, self.neg_items)
        self.u_embeddings_MLP = tf.nn.embedding_lookup(self.user_embeddings_MLP, self.users)
        self.pos_i_embeddings_MLP = tf.nn.embedding_lookup(self.item_embeddings_MLP, self.pos_items)
        self.neg_i_embeddings_MLP = tf.nn.embedding_lookup(self.item_embeddings_MLP, self.neg_items)

        ## logits
        self.pos_scores = self.predict(self.u_embeddings_GMF, self.pos_i_embeddings_GMF, self.u_embeddings_MLP, self.pos_i_embeddings_MLP)
        self.neg_scores = self.predict(self.u_embeddings_GMF, self.neg_i_embeddings_GMF, self.u_embeddings_MLP, self.neg_i_embeddings_MLP)

        ## loss function
        if self.loss_function == 'BPR': self.loss = bpr_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'CrossEntropy': self.loss = cross_entropy_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'MSE': self.loss = mse_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'WBPR': self.loss = wbpr_loss(self.pos_scores, self.neg_scores, self.popularity, self.neg_items)
        if self.loss_function == 'ShiftMC': self.loss = shift_mc_loss(self.pos_scores, self.neg_scores, self.rho)
        if self.loss_function == 'DLNRS':
            self.loss, self.samp_var = dlnrs_loss([self.pos_scores, self.neg_scores],
                                                  [self.sampler, self.lamda, self.aux_loss_weight],
                                                  [self.n_users, self.n_items, self.emb_dim, self.if_pretrain, self.A_hat, self.graph_emb, self.U, self.V],
                                                  [self.users, self.pos_items, self.neg_items])
            self.var_list += self.samp_var

        ## regularization
        self.loss += self.lamda * regularization([self.u_embeddings_GMF, self.pos_i_embeddings_GMF, self.neg_i_embeddings_GMF,
                                                  self.u_embeddings_MLP, self.pos_i_embeddings_MLP, self.neg_i_embeddings_MLP])

        ## optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

        ## get top k
        self.all_ratings = self.get_all_rating(self.u_embeddings_GMF, self.item_embeddings_GMF, self.u_embeddings_MLP, self.item_embeddings_MLP)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def predict(self, user_GMF, item_GMF, user_MLP, item_MLP):
        emb_GMF = tf.multiply(user_GMF, item_GMF)
        emb_MLP = MLP(tf.concat([user_MLP, item_MLP], axis=1), self.W, self.b)
        emb = tf.concat([emb_GMF, emb_MLP], axis=1)
        return tf.reshape(tf.matmul(emb, self.h, transpose_a=False, transpose_b=True), [-1])  # reshpae is not necessary with bpr loss but crutial with cross entropy loss

    def get_all_rating(self, user_GMF, item_GMF, user_MLP, item_MLP):
        n_user_b = tf.shape(user_GMF)[0]
        n_item_b = tf.shape(item_GMF)[0]
        user_GMF_b = tf.reshape(tf.tile(user_GMF, [1, n_item_b]), [-1, int(self.emb_dim/2)])
        item_GMF_b = tf.tile(item_GMF, [n_user_b, 1])
        user_MLP_b = tf.reshape(tf.tile(user_MLP, [1, n_item_b]), [-1, int(self.emb_dim/2)])
        item_MLP_b = tf.tile(item_MLP, [n_user_b, 1])
        score = self.predict(user_GMF_b, item_GMF_b, user_MLP_b, item_MLP_b)
        score = tf.reshape(score, [n_user_b, -1])
        return score
