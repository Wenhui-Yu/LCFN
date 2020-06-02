import tensorflow as tf

class model_BPR(object):
    def __init__(self,n_users,n_items,emb_dim,lr,lamda,optimization):
        self.model_name = 'BPR'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.lamda = lamda
        self.optimization = optimization

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(tf.float32, shape=(None))

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')

        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
#         self.u_embeddings_drop = tf.nn.dropout(self.u_embeddings, self.keep_prob[0])
#         self.pos_i_embeddings_drop = tf.nn.dropout(self.pos_i_embeddings, self.keep_prob[0])
#         self.neg_i_embeddings_drop = tf.nn.dropout(self.neg_i_embeddings, self.keep_prob[0])

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)

#         self.loss = self.create_bpr_loss(self.u_embeddings_drop, self.pos_i_embeddings_drop, self.neg_i_embeddings_drop)
        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings])

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        loss = tf.negative(tf.reduce_sum(maxi)) + self.lamda * regularizer
        return loss

