## with partial dropout
## our paper: self-propagation GNN
## 

import tensorflow as tf
import numpy as np

class model_SGNN(object):
    def __init__(self, layer, n_users, n_items, emb_dim, frequency, lr, lamda, optimization, pre_train_latent_factor,
                 pre_train_propagation_embeddings, if_pretrain):
        self.model_name = 'SGNN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.frequency = frequency
        self.layer = layer
        self.lamda = lamda
        self.lr = lr
        self.optimization = optimization
        [self.U, self.V] = pre_train_latent_factor
        [self.P, self.Q] = pre_train_propagation_embeddings
        self.if_pretrain = if_pretrain

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
            self.user_convolution_base = tf.Variable(self.P, name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(self.Q, name='item_convolution_bases')
        else:
            self.user_embeddings = tf.Variable(
                tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings')
            self.item_embeddings = tf.Variable(
                tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings')
            self.user_convolution_base = tf.Variable(
                tf.random_normal([self.n_users, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(
                tf.random_normal([self.n_items, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='item_convolution_bases')
        self.user_convolution_bases = tf.nn.relu(self.user_convolution_base)
        self.item_convolution_bases = tf.nn.relu(self.item_convolution_base)
        self.decay = tf.Variable(1.0 / (self.n_users+self.n_items), name='decay')
        self.filters_1 = []
        self.filters_2 = []
        for k in range(self.layer): 
            self.filters_1.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
            self.filters_2.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
        
        ## Dropout
        convolution_bases = tf.concat([self.user_convolution_bases, self.item_convolution_bases], axis=0)
        prop_dropout = tf.ones([self.n_users+self.n_items, 1])
        prop_dropout = tf.nn.dropout(prop_dropout, self.keep_prob[0])
        convolution_bases_drop = tf.multiply(convolution_bases, prop_dropout)  # set several rows of convolution_bases to 0
        
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.layer):
            propagated = self.decay * tf.matmul(convolution_bases, tf.matmul(convolution_bases_drop, embeddings, transpose_a=True, transpose_b=False))
            embeddings_1 = propagated + embeddings
            embeddings_2 = tf.multiply(propagated, embeddings)
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings_1, self.filters_1[k]) + tf.matmul(embeddings_2, self.filters_2[k]))
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        self.u_bases = tf.nn.embedding_lookup(self.user_convolution_bases, self.users)
        self.pos_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.pos_items)
        self.neg_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.neg_items)
        
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)

        self.u_embeddings_loss = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
        
        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings) + \
                    self.create_mse_loss(self.u_bases, self.pos_i_bases, self.neg_i_bases) + \
                    self.lamda * self.regularization(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings,
                                                     self.u_bases, self.pos_i_bases, self.neg_i_bases, self.decay)
        
        if self.optimization == 'SGD':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimization == 'RMSProp':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings, self.decay,
                                                              self.user_convolution_base, self.item_convolution_base]
                                                             + self.filters_1 + self.filters_2)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def create_mse_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        loss = tf.nn.l2_loss(1-pos_scores) + tf.nn.l2_loss(neg_scores)
        return loss

    def regularization(self, users, pos_items, neg_items, user_bases, pos_bases, neg_bases, decay):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items) + tf.nn.l2_loss(decay) + \
                      tf.nn.l2_loss(user_bases) + tf.nn.l2_loss(pos_bases) + tf.nn.l2_loss(neg_bases)
        return regularizer





'''
## no dropout
## our paper: self-propagation GNN
## 

import tensorflow as tf
import numpy as np

class model_SGNN(object):
    def __init__(self, layer, n_users, n_items, emb_dim, frequency, lr, lamda, optimization, pre_train_latent_factor,
                 pre_train_propagation_embeddings, if_pretrain):
        self.model_name = 'SGNN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.frequency = frequency
        self.layer = layer
        self.lamda = lamda
        self.lr = lr
        self.optimization = optimization
        [self.U, self.V] = pre_train_latent_factor
        [self.P, self.Q] = pre_train_propagation_embeddings
        self.if_pretrain = if_pretrain

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
            self.user_convolution_base = tf.Variable(self.P, name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(self.Q, name='item_convolution_bases')
        else:
            self.user_embeddings = tf.Variable(
                tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings')
            self.item_embeddings = tf.Variable(
                tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings')
            self.user_convolution_base = tf.Variable(
                tf.random_normal([self.n_users, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(
                tf.random_normal([self.n_items, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='item_convolution_bases')
        self.user_convolution_bases = tf.nn.relu(self.user_convolution_base)
        self.item_convolution_bases = tf.nn.relu(self.item_convolution_base)
        self.decay = tf.Variable(1.0 / (self.n_users+self.n_items), name='decay')
        self.filters_1 = []
        self.filters_2 = []
        for k in range(self.layer): 
            self.filters_1.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
            self.filters_2.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
        
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        convolution_bases = tf.concat([self.user_convolution_bases, self.item_convolution_bases], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.layer):
            propagated = self.decay * tf.matmul(convolution_bases, tf.matmul(convolution_bases, embeddings, transpose_a=True, transpose_b=False))
            embeddings_1 = propagated + embeddings
            embeddings_2 = tf.multiply(propagated, embeddings)
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings_1, self.filters_1[k]) + tf.matmul(embeddings_2, self.filters_2[k]))
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        self.u_bases = tf.nn.embedding_lookup(self.user_convolution_bases, self.users)
        self.pos_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.pos_items)
        self.neg_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.neg_items)

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)

        self.u_embeddings_loss = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)

        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings) + \
                    self.create_mse_loss(self.u_bases, self.pos_i_bases, self.neg_i_bases) + \
                    self.lamda * self.regularization(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings,
                                                     self.u_bases, self.pos_i_bases, self.neg_i_bases, self.decay)

        if self.optimization == 'SGD':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimization == 'RMSProp':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings, self.decay,
                                                              self.user_convolution_base, self.item_convolution_base]
                                                             + self.filters_1 + self.filters_2)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def create_mse_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        loss = tf.nn.l2_loss(1-pos_scores) + tf.nn.l2_loss(neg_scores)
        return loss

    def regularization(self, users, pos_items, neg_items, user_bases, pos_bases, neg_bases, decay):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items) + tf.nn.l2_loss(decay) + \
                      tf.nn.l2_loss(user_bases) + tf.nn.l2_loss(pos_bases) + tf.nn.l2_loss(neg_bases)
        return regularizer
'''



    
    


'''
## with dropout
## our paper: self-propagation GNN
## 

import tensorflow as tf
import numpy as np

class model_SGNN(object):
    def __init__(self, layer, n_users, n_items, emb_dim, frequency, lr, lamda, optimization, pre_train_latent_factor,
                 pre_train_propagation_embeddings, if_pretrain):
        self.model_name = 'SGNN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.frequency = frequency
        self.layer = layer
        self.lamda = lamda
        self.lr = lr
        self.optimization = optimization
        [self.U, self.V] = pre_train_latent_factor
        [self.P, self.Q] = pre_train_propagation_embeddings
        self.if_pretrain = if_pretrain

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
            self.user_convolution_base = tf.Variable(self.P, name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(self.Q, name='item_convolution_bases')
        else:
            self.user_embeddings = tf.Variable(
                tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings')
            self.item_embeddings = tf.Variable(
                tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings')
            self.user_convolution_base = tf.Variable(
                tf.random_normal([self.n_users, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='user_convolution_bases')
            self.item_convolution_base = tf.Variable(
                tf.random_normal([self.n_items, self.frequency], mean=0.0001, stddev=0.00002, dtype=tf.float32),
                name='item_convolution_bases')
        self.user_convolution_bases = tf.nn.relu(self.user_convolution_base)
        self.item_convolution_bases = tf.nn.relu(self.item_convolution_base)
        self.decay = tf.Variable(1.0 / (self.n_users+self.n_items), name='decay')
        self.filters_1 = []
        self.filters_2 = []
        for k in range(self.layer): 
            self.filters_1.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
            self.filters_2.append(tf.Variable((np.random.normal(0, 0.01, (self.emb_dim, self.emb_dim)) + \
                                             np.diag(np.random.normal(1, 0.01, self.emb_dim))).astype(np.float32)))
        
        ## Dropout
        self.filters_1_drop = tf.nn.dropout(self.filters_1, self.keep_prob[3])
        self.filters_2_drop = tf.nn.dropout(self.filters_2, self.keep_prob[3])
        convolution_bases = tf.concat([self.user_convolution_bases, self.item_convolution_bases], axis=0)
        convolution_bases_drop = tf.nn.dropout(convolution_bases, self.keep_prob[1])
        prop_dropout = tf.ones([self.n_users+self.n_items, 1])
        prop_dropout = tf.nn.dropout(prop_dropout, self.keep_prob[2])
        convolution_bases_drop2 = tf.multiply(convolution_bases_drop, prop_dropout)  # set several rows of convolution_bases to 0
        
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.layer):
            propagated = self.decay * tf.matmul(convolution_bases_drop, tf.matmul(convolution_bases_drop2, embeddings, transpose_a=True, transpose_b=False))
            embeddings_1 = propagated + embeddings
            embeddings_2 = tf.multiply(propagated, embeddings)
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings_1, self.filters_1_drop[k]) + tf.matmul(embeddings_2, self.filters_2_drop[k]))
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)
        self.u_embeddings_drop = tf.nn.dropout(self.u_embeddings, self.keep_prob[0])
        self.pos_i_embeddings_drop = tf.nn.dropout(self.pos_i_embeddings, self.keep_prob[0])
        self.neg_i_embeddings_drop = tf.nn.dropout(self.neg_i_embeddings, self.keep_prob[0])

        self.u_bases = tf.nn.embedding_lookup(self.user_convolution_bases, self.users)
        self.pos_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.pos_items)
        self.neg_i_bases = tf.nn.embedding_lookup(self.item_convolution_bases, self.neg_items)
        self.u_bases_drop = tf.nn.dropout(self.u_bases, self.keep_prob[1])
        self.pos_i_bases_drop = tf.nn.dropout(self.pos_i_bases, self.keep_prob[1])
        self.neg_i_bases_drop = tf.nn.dropout(self.neg_i_bases, self.keep_prob[1])
        

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)

        self.u_embeddings_loss = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_loss = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
        

        self.loss = self.create_bpr_loss(self.u_embeddings_drop, self.pos_i_embeddings_drop, self.neg_i_embeddings_drop) + \
                    self.create_mse_loss(self.u_bases_drop, self.pos_i_bases_drop, self.neg_i_bases_drop) + \
                    self.lamda * self.regularization(self.u_embeddings_drop, self.pos_i_embeddings_drop, self.neg_i_embeddings_drop,
                                                     self.u_bases_drop, self.pos_i_bases_drop, self.neg_i_bases_drop, self.decay)
        
        if self.optimization == 'SGD':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimization == 'RMSProp':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings, self.decay,
                                                              self.user_convolution_base, self.item_convolution_base]
                                                             + self.filters_1 + self.filters_2)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def create_mse_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        loss = tf.nn.l2_loss(1-pos_scores) + tf.nn.l2_loss(neg_scores)
        return loss

    def regularization(self, users, pos_items, neg_items, user_bases, pos_bases, neg_bases, decay):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items) + tf.nn.l2_loss(decay) + \
                      tf.nn.l2_loss(user_bases) + tf.nn.l2_loss(pos_bases) + tf.nn.l2_loss(neg_bases)
        return regularizer

'''