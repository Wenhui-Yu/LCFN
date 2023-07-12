## basic baseline MF_BPR

import tensorflow as tf

class model_MF(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'MF'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.popularity = data['popularity']
        self.A_hat = data['sparse_propagation_matrix']
        self.graph_emb = data['graph_embeddings']

        ## placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        ## define trainable parameters
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
        self.var_list = [self.user_embeddings, self.item_embeddings]

        ## lookup
        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)

        ## logits
        self.pos_scores = self.inner_product(self.u_embeddings, self.pos_i_embeddings)
        self.neg_scores = self.inner_product(self.u_embeddings, self.neg_i_embeddings)

        ## loss function
        if self.loss_function == 'BPR': self.loss = self.bpr_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'CrossEntropy': self.loss = self.cross_entropy_loss(self.pos_scores, self.neg_scores)

        ## regularization
        self.loss += self.lamda * self.regularization([self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings])

        ## optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

        ## get top k
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def inner_product(self, users, items):
        scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
        return scores

    def bpr_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def cross_entropy_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores)) + tf.log(1 - tf.nn.sigmoid(neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list: reg += tf.nn.l2_loss(para)
        return reg
