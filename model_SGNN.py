## our model: Self-propagation Graph Neural Network (SGNN)
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import tensorflow as tf

class model_SGNN(object):
    def __init__(self, n_users, n_items, lr, lamda, emb_dim, layer, pre_train_latent_factor, propagation_embeddings,
                 if_pretrain, prop_emb):
        self.model_name = 'SGNN'
        self.n_users = n_users
        self.n_items = n_items
        ## hyperparameters
        self.lr = lr
        self.lamda = lamda
        self.emb_dim = emb_dim
        if prop_emb == 'RM': self.prop_dim = propagation_embeddings[0].shape[1]
        self.layer = layer
        ## model parameters
        self.U, self.V = pre_train_latent_factor
        if prop_emb == 'RM': [self.P, self.Q] = propagation_embeddings
        if prop_emb == 'SF': self.propagation_sf = propagation_embeddings
        self.if_pretrain = if_pretrain
        self.prop_emb = prop_emb
        self.layer_weight = [1 / (l + 1) ** 1 for l in range(self.layer + 1)]

        ## placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        ## learnable parameters
        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
            if self.prop_emb == 'RM':
                self.user_propagation = tf.Variable(self.P, name='user_propagation')
                self.item_propagation = tf.Variable(self.Q, name='item_propagation')
        else:
            self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
            self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
            if self.prop_emb == 'RM':
                self.user_propagation = tf.Variable(tf.random_normal([self.n_users, self.prop_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_propagation')
                self.item_propagation = tf.Variable(tf.random_normal([self.n_items, self.prop_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_propagation')
        if self.prop_emb != 'SF': self.normalization = tf.Variable(1 / (self.n_users + self.n_items), name='normalization')

        ## propagation layers definition
        self.embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        if self.prop_emb == 'RM': self.propagation_rm = tf.concat([self.user_propagation, self.item_propagation], axis=0)
        self.all_embeddings = self.embeddings
        for l in range(self.layer):
            ## low-pass graph convolution
            if self.prop_emb == 'RM': self.embeddings = tf.matmul(self.propagation_rm, tf.matmul(self.propagation_rm, self.embeddings, transpose_a=True, transpose_b=False))
            if self.prop_emb == 'SF': self.embeddings = tf.matmul(self.propagation_sf, tf.matmul(self.propagation_sf, self.embeddings, transpose_a=True, transpose_b=False))
            if self.prop_emb == 'PE': self.embeddings = tf.matmul(self.all_embeddings, tf.matmul(self.all_embeddings, self.embeddings, transpose_a=True, transpose_b=False))
            if self.prop_emb == 'SF': self.all_embeddings += self.layer_weight[l + 1] * tf.nn.tanh(self.embeddings)
            else: self.all_embeddings += self.layer_weight[l + 1] * tf.nn.tanh(self.normalization * self.embeddings)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(self.all_embeddings, [self.n_users, self.n_items], 0)

        ## make prediction
        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        if self.prop_emb == 'RM':
            self.u_propagation = tf.nn.embedding_lookup(self.user_propagation, self.users)
            self.pos_i_propagation = tf.nn.embedding_lookup(self.item_propagation, self.pos_items)
            self.neg_i_propagation = tf.nn.embedding_lookup(self.item_propagation, self.neg_items)

        ## generalization
        self.u_embeddings_reg = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
        self.reg_list = [self.u_embeddings_reg, self.pos_i_embeddings_reg, self.neg_i_embeddings_reg]

        ## loss function and updating
        self.loss = self.bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings)
        self.var_list = [self.user_embeddings, self.item_embeddings]    ## learnable parameter list
        if self.prop_emb != 'SF': self.var_list += [self.normalization]
        if self.prop_emb == 'RM':
            self.loss += self.mse_loss(self.u_propagation, self.pos_i_propagation, self.neg_i_propagation)
            self.var_list += [self.user_propagation, self.item_propagation]
            self.u_propagation_reg = tf.nn.embedding_lookup(self.user_propagation, self.users)
            self.pos_i_propagation_reg = tf.nn.embedding_lookup(self.item_propagation, self.pos_items)
            self.neg_i_propagation_reg = tf.nn.embedding_lookup(self.item_propagation, self.neg_items)
            self.reg_list += [self.u_propagation_reg, self.pos_i_propagation_reg, self.neg_i_propagation_reg]
        self.loss += self.lamda * self.regularization(self.reg_list)

        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

        ## prediction
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings += self.items_in_train_data    ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def mse_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        loss = tf.nn.l2_loss(1 - pos_scores) + tf.nn.l2_loss(neg_scores)
        return loss

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list: reg += tf.nn.l2_loss(para)
        return reg