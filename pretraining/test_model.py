## author@Wenhui Yu  2021.02.16
## test the model and return the performance

from numpy import *
import numpy as np
import random as rd

def test_one_user(user, top_item, para_test_one_user):
    [test_data, TOP_K] = para_test_one_user
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    top_item = top_item.tolist()  ## make testing fatser
    for i in range(k_num):
        f1[i] = evaluation_F1(top_item, TOP_K[i], test_data[user])
        ndcg[i] = evaluation_NDCG(top_item, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model, para_test):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    para_test_one_user = [test_data, TOP_K]
    # test a batch user instead of whole user
    user_top_items = np.zeros((TEST_USER_BATCH, max(TOP_K))).astype(dtype=int32)
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num))
    mini_batch_list.append(TEST_USER_BATCH)
    score_min = -10 ** 5
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1]
        user_batch = test_batch[u1: u2]
        items_in_train_data = np.zeros((u2 - u1, item_num))
        for u_index, user in enumerate(user_batch):
            for item in train_data[user]:
                items_in_train_data[u_index, item] = score_min
        user_top_items_batch = sess.run(model.top_items, feed_dict={model.users: user_batch, model.items_in_train_data: items_in_train_data, model.top_k: max(TOP_K)})
        user_top_items[u1: u2] = user_top_items_batch
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user(user, user_top_items[u_index], para_test_one_user))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    return F1, NDCG

def evaluation_F1(order, top_k, positive_item):
    epsilon = 0.1 ** 10
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / max(len(top_k_items), epsilon)
    recall = len(top_k_items & positive_item) / max(len(positive_item), epsilon)
    F1 = 2 * precision * recall / max(precision + recall, epsilon)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    epsilon = 0.1**10
    DCG = 0
    iDCG = 0
    for i in range(top_k):
        if top_k_item[i] in positive_item:
            DCG += 1 / log2(i + 2)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / log2(i + 2)
    NDCG = DCG / max(iDCG, epsilon)
    return NDCG