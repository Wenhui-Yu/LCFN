## graph embeddings as the Fourier transform bases for LGCN model
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import scipy as sp
import scipy.sparse.linalg
from numpy import *
import json

DATASET = 0             # 0 for Amazon, 1 for Movielens
FREQUENCY = 128         # dimensionality of the base
FREQUENCY_U = [100, 300][DATASET]   # dimensionality of the base of the user graph
FREQUENCY_I = [50, 200][DATASET]    # dimensionality of the base of the user graph
GRAPH_CONV = ['1d', '2d'][0]            # 0 for 1d convolution and 1 for 2d
Dataset = ['Amazon', 'Movielens'][DATASET]
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10

root = '../dataset/'
path_train = root + Dataset + '/train_data.json'
path_save = root + Dataset + '/graph_embeddings_' + GRAPH_CONV + '.json'
print('Reading data...')
with open(path_train) as f:
    line = f.readline()
    data = json.loads(line)
f.close()

user_number = len(data)
item_number = 0
for item_list in data: item_number = max(item_number, max(item_list))
item_number += 1

if GRAPH_CONV == '1d':
    print('Initializing...')
    A = sp.sparse.lil_matrix((user_number + item_number, user_number + item_number))
    D = sp.sparse.lil_matrix((user_number + item_number, user_number + item_number))
    I = sp.sparse.lil_matrix((user_number + item_number, user_number + item_number))
    for i in range(user_number + item_number): I[i, i] = 1

    # constructing the laplacian matrices
    print('Constructing the laplacian matrices...')
    for u in range(user_number):
        for i in data[u]:
            A[u, user_number + i] = 1
            A[user_number + i, u] = 1
            D[u, u] += 1
            D[user_number + i, user_number + i] += 1
    for l in range(user_number + item_number):
        D[l, l] = 1.0 / max(sqrt(D[l, l]), epsilon)
    L = I - D * A * D

    #eigenvalue factorization
    print('Decomposing the laplacian matrices...')
    [Lamda, graph_embeddings] = sp.sparse.linalg.eigsh(L, k = FREQUENCY, which='SM', tol = tolerant)
    print(Lamda[0:10])

    print('Saving features...')
    f = open(path_save, 'w')
    jsObj = json.dumps(graph_embeddings.tolist())
    f.write(jsObj)
    f.close()
else:
    print('Constructing user-user data and item-item data...')
    items_of_user = data
    users_of_item = []
    for i in range(item_number): users_of_item.append([])
    for u, item_list in enumerate(items_of_user):
        for i in item_list:
            users_of_item[i].append(u)
    user_interaction = {}
    item_interaction = {}
    for user_list in users_of_item:
        for u in range(len(user_list)):
            for v in range(u, len(user_list)):
                key = (user_list[u], user_list[v]) if user_list[u] < user_list[v] else (user_list[v], user_list[u])
                user_interaction[key] = 1
    for item_list in items_of_user:
        for i in range(len(item_list)):
            for j in range(i, len(item_list)):
                key = (item_list[i], item_list[j]) if item_list[i] < item_list[j] else (item_list[j], item_list[i])
                item_interaction[key] = 1

    print('Initializing...')
    A_user = sp.sparse.lil_matrix((user_number, user_number))
    D_user = sp.sparse.lil_matrix((user_number, user_number))
    I_user = sp.sparse.lil_matrix((user_number, user_number))
    for u in range(user_number): I_user[u, u] = 1
    A_item = sp.sparse.lil_matrix((item_number, item_number))
    D_item = sp.sparse.lil_matrix((item_number, item_number))
    I_item = sp.sparse.lil_matrix((item_number, item_number))
    for i in range(item_number): I_item[i, i] = 1

    print('Constructing the laplacian matrices...')
    print('   constructing the user matrix...')
    for u, v in user_interaction:
        A_user[u, v] = 1
        A_user[v, u] = 1
        D_user[u, u] += 1
        if u != v: D_user[v, v] += 1
    print('   constructing the item matrix...')
    for i, j in item_interaction:
        A_item[i, j] = 1
        A_item[j, i] = 1
        D_item[i, i] += 1
        if i != j: D_item[j, j] += 1
    print('   constructing the degree matrix...')
    for u in range(user_number):
        D_user[u, u] = 1.0 / max(sqrt(D_user[u, u]), epsilon)
    L_user = I_user - D_user * A_user * D_user
    for i in range(item_number):
        D_item[i, i] = 1.0 / max(sqrt(D_item[i, i]), epsilon)
    L_item = I_item - D_item * A_item * D_item

    print('Decomposing the laplacian matrices...')
    print('   decomposing user matrix...')
    [Lamda, graph_embeddings_user] = sp.sparse.linalg.eigsh(L_user, k=FREQUENCY_U, which='SM', tol=tolerant)
    print(Lamda[0:10])
    print('   decomposing item matrix...')
    [Lamda, graph_embeddings_item] = sp.sparse.linalg.eigsh(L_item, k=FREQUENCY_I, which='SM', tol=tolerant)
    print(Lamda[0:10])

    print('Saving features...')
    f = open(path_save, 'w')
    jsObj = json.dumps([graph_embeddings_user.tolist(), graph_embeddings_item.tolist()])
    f.write(jsObj)
    f.close()