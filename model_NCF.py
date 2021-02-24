## baseline: Neural Collaborative Filtering (NCF)
## Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web, WWW '17, pages 173-182, 2017.

import tensorflow as tf
import numpy as np

class model_NCF(object):
    def __init__(self, layer, n_users, n_items, emb_dim, lr, lamda, pre_train_latent_factor,if_pretrain):
        self.model_name = 'NCF'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda
        self.layer = layer
        [self.U, self.V] = pre_train_latent_factor
        self.if_pretrain = if_pretrain
        self.weight_size_list = [2 * self.emb_dim]
        for l in range(self.layer):
            self.weight_size_list.append(max(int(0.5 ** l * 64), 4))

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))
        self.items_in_train_data = tf.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.placeholder(tf.int32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings_GMF = tf.Variable(self.U, name='user_embeddings_GMF')
            self.item_embeddings_GMF = tf.Variable(self.V, name='item_embeddings_GMF')
            self.user_embeddings_MLP = tf.Variable(self.U, name='user_embeddings_MLP')
            self.item_embeddings_MLP = tf.Variable(self.V, name='item_embeddings_MLP')
        else:
            self.user_embeddings_GMF = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings_GMF')
            self.item_embeddings_GMF = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_GMF')
            self.user_embeddings_MLP = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings_MLP')
            self.item_embeddings_MLP = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_MLP')
        self.W = []
        self.b = []
        for l in range(self.layer):
            self.W.append(tf.Variable(tf.random_normal([self.weight_size_list[l], self.weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
            self.b.append(tf.Variable(tf.random_normal([1, self.weight_size_list[l + 1]], mean=0.01, stddev=0.02, dtype=tf.float32)))
        self.h = tf.Variable(tf.random_normal([1, self.emb_dim + self.weight_size_list[-1]], mean=0.01, stddev=0.02, dtype=tf.float32), name='h')

        self.u_embeddings_GMF = tf.nn.embedding_lookup(self.user_embeddings_GMF, self.users)
        self.pos_i_embeddings_GMF = tf.nn.embedding_lookup(self.item_embeddings_GMF, self.pos_items)
        self.neg_i_embeddings_GMF = tf.nn.embedding_lookup(self.item_embeddings_GMF, self.neg_items)
        self.u_embeddings_MLP = tf.nn.embedding_lookup(self.user_embeddings_MLP, self.users)
        self.pos_i_embeddings_MLP = tf.nn.embedding_lookup(self.item_embeddings_MLP, self.pos_items)
        self.neg_i_embeddings_MLP = tf.nn.embedding_lookup(self.item_embeddings_MLP, self.neg_items)

        self.pos_ratings = self.predict(self.u_embeddings_GMF, self.pos_i_embeddings_GMF, self.u_embeddings_MLP, self.pos_i_embeddings_MLP)
        self.neg_ratings = self.predict(self.u_embeddings_GMF, self.neg_i_embeddings_GMF, self.u_embeddings_MLP, self.neg_i_embeddings_MLP)
        self.loss = self.create_bpr_loss(self.pos_ratings, self.neg_ratings)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings_GMF, self.item_embeddings_GMF, self.h,
                                                              self.user_embeddings_MLP, self.item_embeddings_MLP] + self.W + self.b)

        self.all_ratings = self.get_all_rating(self.u_embeddings_GMF, self.item_embeddings_GMF, self.u_embeddings_MLP, self.item_embeddings_MLP)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def create_bpr_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        return tf.negative(tf.reduce_sum(maxi))
    
    def regularization(self, Paras):
        regularizer = 0
        for para in Paras:
            regularizer += tf.nn.l2_loss(para)
        return regularizer
    
    def GMF(self, use_emb, item_emb):
        emb = tf.multiply(use_emb, item_emb)
        return emb
    
    def MLP(self, use_emb, item_emb):
        emb = tf.concat([use_emb, item_emb], axis=1)
        for l in range(self.layer):
            emb = tf.nn.relu(tf.matmul(emb, self.W[l]) + self.b[l])
        return emb

    def predict(self, user_GMF, item_GMF, user_MLP, item_MLP):
        emb_GMF = self.GMF(user_GMF, item_GMF)
        emb_MLP = self.MLP(user_MLP, item_MLP)
        emb = tf.concat([emb_GMF, emb_MLP], axis=1)
        # return tf.nn.sigmoid(tf.reshape(tf.matmul(emb, self.h, transpose_a=False, transpose_b=True), [-1]))   # there is a sigmoid in BPR thus we do not use sigmoid here
        return tf.reshape(tf.matmul(emb, self.h, transpose_a=False, transpose_b=True), [-1])   # reshpae is not necessary with bpr loss but crutial with cross entropy loss

    def get_all_rating(self, user_GMF, item_GMF, user_MLP, item_MLP):
        n_user_b = tf.shape(user_GMF)[0]
        n_item_b = tf.shape(item_GMF)[0]
        user_GMF_b = tf.reshape(tf.tile(user_GMF, [1, n_item_b]), [-1, self.emb_dim])
        item_GMF_b = tf.tile(item_GMF, [n_user_b, 1])
        user_MLP_b = tf.reshape(tf.tile(user_MLP, [1, n_item_b]), [-1, self.emb_dim])
        item_MLP_b = tf.tile(item_MLP, [n_user_b, 1])
        score = self.predict(user_GMF_b, item_GMF_b, user_MLP_b, item_MLP_b)
        score = tf.reshape(score, [n_user_b, -1])
        return score
