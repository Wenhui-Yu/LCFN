## author@Wenhui Yu  2019.04.14
## test the model and return the performance

from evaluation import *
from read_data import *
from params import DIR
from params import TOP_K
from params import TEST_VALIDATION
from params import TEST_USER_BATCH
import operator
import random as rd
import multiprocessing
import gc
cores = multiprocessing.cpu_count()

train_path = DIR+'train_data.json'
teat_path = DIR+'test_data.json'
validation_path = DIR+'validation_data.json'

## load data
[train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
teat_vali_path = validation_path if operator.eq(TEST_VALIDATION,'Validation')==1 else teat_path
test_data = read_data(teat_vali_path)[0]
score_min = -10 ** 5
def test_one_user(x):
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    user = x[0]
    score = x[1]
    for item in train_data[user]:
        score[item] = score_min
    order = list(np.argsort(score))
    order.reverse()
    order = order[0: max(TOP_K)]
    for i in range(k_num):
        f1[i] += evaluation_F1(order, TOP_K[i], test_data[user])
        ndcg[i] += evaluation_NDCG(order, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model):
    ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
    ## For some models (NCF), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
    user_score = np.zeros((TEST_USER_BATCH, item_num))
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num))
    mini_batch_list.append(TEST_USER_BATCH)
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1]
        user_batch = test_batch[u1: u2]
        user_score_batch = sess.run(model.all_ratings, feed_dict={model.users: user_batch,
                                                                  model.keep_prob: np.array([1,1,1,1,1])})
        user_score[u1: u2] = user_score_batch
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user([user, user_score[u_index]]))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    del result, user_score_batch, user_score
    gc.collect()
    return F1, NDCG


# def test_model(sess, model):
#     ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
#     ## For some models (NCF), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
#     user_score = np.zeros((TEST_USER_BATCH, item_num))
#     Test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
#     ## remove the users without records in test/validation set
#     test_batch = []
#     for user in Test_batch:
#         if len(test_data[user]) > 0:
#             test_batch.append(user)
#     test_batch = np.array(test_batch)
#     mini_batch_num = 100
#     mini_batch_list = list(range(0, len(test_batch), mini_batch_num))
#     mini_batch_list.append(len(test_batch))
#     for u in range(len(mini_batch_list) - 1):
#         u1, u2 = mini_batch_list[u], mini_batch_list[u + 1]
#         user_batch = test_batch[u1: u2]
#         user_score_batch = sess.run(model.all_ratings, feed_dict={model.users: user_batch})
#         user_score[u1: u2] = user_score_batch
#     ## test by multiprocessing
#     pool = multiprocessing.Pool(cores)
#     user_id_rating = zip(test_batch, user_score)
#     result = pool.map(test_one_user, user_id_rating)
#     pool.close()
#     F1, NDCG = np.mean(np.array(result), axis=0)
#     return F1, NDCG