## our model: Low-pass Graph Convolutional Network (LGCN)
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import tensorflow as tf

class model_LGCN(object):
    def __init__(self, n_users, n_items, lr, lamda, emb_dim, layer, pre_train_latent_factor, graph_embeddings, graph_conv,
                 prediction, loss_function, generalization, optimization, if_pretrain, if_transformation, activation, pooling):
        self.model_name = 'LGCN'
        self.n_users = n_users
        self.n_items = n_items
        ## hyperparameters
        self.lr = lr
        self.lamda = lamda
        self.emb_dim = emb_dim
        self.emb_dim_predict = (layer + 1) * emb_dim if pooling == 'Concat' else emb_dim
        if graph_conv == '1D': self.frequency = graph_embeddings.shape[1]
        else: self.frequency_U, self.frequency_V = graph_embeddings[0].shape[1], graph_embeddings[1].shape[1]
        self.layer = layer
        ## model parameters
        self.U, self.V = pre_train_latent_factor
        if graph_conv == '1D': self.graph_emb = graph_embeddings
        else: self.graph_emb_U, self.graph_emb_V = graph_embeddings
        ## network structure; model settings; and optimization setting
        self.graph_conv = graph_conv
        self.prediction = prediction
        self.loss_function = loss_function
        self.generalization = generalization.split('+')
        self.optimization = optimization
        self.if_pretrain = if_pretrain
        self.if_transformation = if_transformation
        self.activation = activation
        self.pooling = pooling

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
        else:
            self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
            self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
        if graph_conv == '1D': self.kernel = [tf.Variable(tf.random_normal([self.frequency], mean=0.01, stddev=0.02, dtype=tf.float32)) for l in range(self.layer)]
        else: self.kernel_U, self.kernel_V = [tf.Variable(tf.random_normal([self.frequency_U], mean=0.01, stddev=0.02, dtype=tf.float32)) for l in range(self.layer)], [tf.Variable(tf.random_normal([self.frequency_V], mean=0.01, stddev=0.02, dtype=tf.float32)) for l in range(self.layer)]
        if self.if_transformation: self.transformation = [tf.Variable(tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32)) for l in range(self.layer)]
        if self.pooling == 'Sum': self.layer_weight = [(1 / (l + 1)) ** 1 for l in range(self.layer + 1)]
        if self.pooling[0: 3] == 'MLP':
            self.pooling_mlp_layer = int(self.pooling[3:])
            self.pooling_layer_size = [(self.layer + 1) * self.emb_dim] + [(self.pooling_mlp_layer - l) * self.emb_dim_predict for l in range(self.pooling_mlp_layer)]
            self.pooling_W = [tf.Variable(tf.random_normal([self.pooling_layer_size[l], self.pooling_layer_size[l + 1]], mean=0, stddev=0.01, dtype=tf.float32)) for l in range(self.pooling_mlp_layer)]
            self.pooling_b = [tf.Variable(tf.random_normal([self.pooling_layer_size[l + 1]], mean=0, stddev=0.01, dtype=tf.float32)) for l in range(self.pooling_mlp_layer)]
        if self.prediction[0: 3] == 'MLP':
            self.prediction_mlp_layer = int(self.prediction[3:])
            self.prediction_layer_size = [3 * self.emb_dim_predict] + [self.emb_dim_predict] * (self.prediction_mlp_layer - 1) + [1]
            self.prediction_W = [tf.Variable(tf.random_normal([self.prediction_layer_size[l], self.prediction_layer_size[l + 1]], mean=0, stddev=0.01, dtype=tf.float32)) for l in range(self.prediction_mlp_layer)]
            self.prediction_b = [tf.Variable(tf.random_normal([self.prediction_layer_size[l + 1]], mean=0, stddev=0.01, dtype=tf.float32)) for l in range(self.prediction_mlp_layer)]

        ## convolutional layers definition
        self.embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        if self.pooling in ['Sum', 'Product']: self.all_embeddings = self.embeddings
        else: self.all_embeddings = [self.embeddings]
        for l in range(self.layer):
            ## low-pass graph convolution
            if self.graph_conv == '1D': self.embeddings = tf.matmul(tf.matmul(self.graph_emb, tf.diag(self.kernel[l])), tf.matmul(self.graph_emb, self.embeddings, transpose_a=True, transpose_b=False))
            else:
                self.embeddings_U, self.embeddings_V = tf.split(self.embeddings, [self.n_users, self.n_items], axis=0)
                self.embeddings_U = tf.matmul(tf.matmul(self.graph_emb_U, tf.diag(self.kernel_U[l])), tf.matmul(self.graph_emb_U, self.embeddings_U, transpose_a=True, transpose_b=False))
                self.embeddings_V = tf.matmul(tf.matmul(self.graph_emb_V, tf.diag(self.kernel_V[l])), tf.matmul(self.graph_emb_V, self.embeddings_V, transpose_a=True, transpose_b=False))
                self.embeddings = tf.concat([self.embeddings_U, self.embeddings_V], axis=0)
            if self.if_transformation: self.embeddings = tf.matmul(self.embeddings, self.transformation[l])
            ## activations and pooling
            if self.activation == 'Sigmoid': self.embeddings = tf.nn.sigmoid(self.embeddings)
            if self.activation == 'Tanh': self.embeddings = tf.nn.tanh(self.embeddings)
            if self.activation == 'ReLU': self.embeddings = tf.nn.relu(self.embeddings)
            if self.pooling == 'Sum': self.all_embeddings += self.embeddings * self.layer_weight[l + 1]
            elif self.pooling == 'Product': self.all_embeddings = tf.multiply(self.all_embeddings, tf.nn.sigmoid(self.embeddings))  ## product makes -- to + and confuses the model, thus needs to map the value to + first
            else: self.all_embeddings += [self.embeddings]  ##concat, max, and mlp

        ## pooling to get predictive embeddings
        if self.pooling == 'Concat': self.all_embeddings = tf.concat(self.all_embeddings, 1)
        if self.pooling == 'Max': self.all_embeddings = tf.reduce_max(self.all_embeddings, 0)
        if self.pooling[0: 3] == 'MLP': self.all_embeddings = tf.nn.tanh(self.MLP(tf.concat(self.all_embeddings, 1), self.pooling_W, self.pooling_b))
        if 'L2Norm' in self.generalization: self.all_embeddings = tf.nn.l2_normalize(self.all_embeddings, 1)
        self.user_all_embeddings, self.item_all_embeddings = tf.split(self.all_embeddings, [self.n_users, self.n_items], 0)

        ## make prediction
        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)
        if 'DropOut' in self.generalization:
            self.u_embeddings = tf.nn.dropout(self.u_embeddings, self.keep_prob)
            self.pos_i_embeddings = tf.nn.dropout(self.pos_i_embeddings, self.keep_prob)
            self.neg_i_embeddings = tf.nn.dropout(self.neg_i_embeddings, self.keep_prob)
        if self.prediction == 'InnerProduct':
            self.pos_scores = tf.reduce_sum(tf.multiply(self.u_embeddings, self.pos_i_embeddings), 1)
            self.neg_scores = tf.reduce_sum(tf.multiply(self.u_embeddings, self.neg_i_embeddings), 1)
            self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)
        else:
            self.pos_scores = self.MLP(tf.concat([self.u_embeddings, self.pos_i_embeddings, tf.multiply(self.u_embeddings, self.pos_i_embeddings)], 1), self.prediction_W, self.prediction_b)
            self.neg_scores = self.MLP(tf.concat([self.u_embeddings, self.neg_i_embeddings, tf.multiply(self.u_embeddings, self.neg_i_embeddings)], 1), self.prediction_W, self.prediction_b)
            self.all_ratings = self.get_all_ratings(self.u_embeddings, self.item_all_embeddings, self.prediction_W, self.prediction_b)
        self.all_ratings += self.items_in_train_data   ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

        ## generalization
        if 'Regularization' in self.generalization:
            self.u_embeddings_reg = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.pos_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.neg_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
            self.reg_list = [self.u_embeddings_reg, self.pos_i_embeddings_reg, self.neg_i_embeddings_reg]
            # if self.graph_conv == '1D': self.reg_list += self.kernel
            # else: self.reg_list += self.kernel_U + self.kernel_V
            # if self.if_transformation: self.reg_list += self.transformation
            # if self.pooling[0: 3] == 'MLP': self.reg_list += self.pooling_W
            # if self.prediction[0: 3] == 'MLP': self.reg_list += self.prediction_W

        ## loss function
        if self.loss_function == 'BPR': self.loss = self.bpr_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'CrossEntropy': self.loss = self.cross_entropy_loss(self.pos_scores, self.neg_scores)
        if self.loss_function == 'MSE': self.loss = self.mse_loss(self.pos_scores, self.neg_scores)
        if 'Regularization' in self.generalization: self.loss += self.lamda * self.regularization(self.reg_list)

        ## optimizer
        if self.optimization == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimization == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimization == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.var_list = [self.user_embeddings, self.item_embeddings]  ## learnable parameter list
        if self.graph_conv == '1D': self.var_list += self.kernel
        else: self.var_list += self.kernel_U + self.kernel_V
        if self.if_transformation: self.var_list += self.transformation
        if self.pooling[0: 3] == 'MLP': self.var_list += self.pooling_W + self.pooling_b
        if self.prediction[0: 3] == 'MLP': self.var_list += self.prediction_W + self.prediction_b
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

    def bpr_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def cross_entropy_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores)) + tf.log(1 - tf.nn.sigmoid(neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def mse_loss(self, pos_scores, neg_scores):
        loss = tf.nn.l2_loss(1 - pos_scores) + tf.nn.l2_loss(neg_scores)
        return loss

    def MLP(self, x, W, b):
        for l in range(len(W) - 1):
            x = tf.nn.tanh(tf.matmul(x, W[l]) + b[l])
            if 'DropOut' in self.generalization:
                x = tf.nn.dropout(x, self.keep_prob)
        return tf.matmul(x, W[-1]) + b[-1]    ## do not perform activation on the output

    def get_all_ratings(self, user_emb, item_emb, W, b):
        user_num = tf.shape(user_emb)[0]
        item_num = tf.shape(item_emb)[0]
        user_emb_extend = tf.reshape(tf.tile(user_emb, [1, item_num]), [-1, self.emb_dim_predict])
        item_emb_extend = tf.tile(item_emb, [user_num, 1])
        score = self.MLP(tf.concat([user_emb_extend, item_emb_extend, tf.multiply(user_emb_extend, item_emb_extend)], 1), W, b)
        score = tf.reshape(score, [user_num, -1])
        return score

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list: reg += tf.nn.l2_loss(para)
        return reg

