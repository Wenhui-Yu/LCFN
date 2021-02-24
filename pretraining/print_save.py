## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

import json
def print_params(para_name, para):
    for i in range(len(para)):
        print(para_name[i]+':  ',para[i])

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

