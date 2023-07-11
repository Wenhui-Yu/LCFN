## author@ Wenhui Yu email: jianlin.ywh@alibaba-inc.com  2021.02.16
## constructing the sparse graph

import tensorflow as tf
import numpy as np

def propagation_matrix(graph, user_num, item_num, norm):
    print('Constructing the sparse graph...')
    eps = 0.1 ** 10
    user_itemNum = np.zeros(user_num)
    item_userNum = np.zeros(item_num)
    for (user, item) in graph:
        user_itemNum[user] += 1
        item_userNum[item] += 1
    val, idx = [], []
    for (user, item) in graph:
        if norm == 'left_norm':
            val += [1 / max(user_itemNum[user], eps), 1 / max(item_userNum[item], eps)]
            idx += [[user, item + user_num], [item + user_num, user]]
        if norm == 'sym_norm':
            val += [1 / (max(np.sqrt(user_itemNum[user] * item_userNum[item]), eps))] * 2
            idx += [[user, item + user_num], [item + user_num, user]]
    return tf.SparseTensor(indices=idx, values=val, dense_shape=[user_num + item_num, user_num + item_num])

