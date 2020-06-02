## author@Wenhui Yu  2019.04.14
## test the model and return the performance

from p_evaluation import *
from p_read_data import *
from p_params import DIR
from p_params import TOP_K
from p_params import TEST_VALIDATION
import operator
import random as rd
import time
#import multiprocessing
#cores = multiprocessing.cpu_count()

train_path = DIR+'train_data.json'
teat_path = DIR+'test_data.json'
validation_path = DIR+'validation_data.json'

## load data
[train_data, train_data_interaction, user_num, item_num] = read_data(train_path,1)
teat_vali_path = validation_path if operator.eq(TEST_VALIDATION,'Validation')==1 else teat_path
test_data = read_data(teat_vali_path,1)[0]

def test_one_user(x):
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    user = x[0]
    score = x[1]
    order = list(np.argsort(score))
    order.reverse()
    order = order[0: max(TOP_K) + len(train_data[user])]
    for item in train_data[user]:
        try: order.remove(item)
        except: continue
    for i in range(k_num):
        f1[i] += evaluation_F1(order, TOP_K[i], test_data[user])
        ndcg[i] += evaluation_NDCG(order, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model):
    test_batch = rd.sample(list(range(user_num)), 4096)
    user_score = sess.run(model.all_ratings, feed_dict={model.users: test_batch})
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user([user, user_score[u_index]]))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    return F1, NDCG