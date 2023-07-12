## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

import json
import random as rd
def print_params(para):
    for para_name in para:
        print(para_name+':  ',para[para_name])

def print_value(value):
    [inter, loss, f1_max, F1, NDCG] = value
    print('iter: %d loss %.2f f1 %.4f' %(inter, loss, f1_max), end='  ')
    print(F1, NDCG)

def save_embeddings(data, path):
    f = open(path, 'w')
    js = json.dumps(data)
    f.write(js)
    f.write('\n')
    f.close

def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return data, interactions, user_num, item_num