## Hypergraph embeddings as the Fourier transform bases for LCFN model
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import scipy as sp
import scipy.sparse.linalg
from numpy import *
import numpy as np
import json

DATASET = 0                             # 0 for Amazon, 1 for Movielens
FREQUENCY_U = [100, 300][DATASET]       # dimensionality of the base of the user graph
FREQUENCY_I = [50, 200][DATASET]        # dimensionality of the base of the user graph
IF_WEIGHTED = [False, True][0]         # 0 for uniform weighted, 1 for weighted by the 1/popularity.
Dataset = ['Amazon', 'Movielens'][DATASET]
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10

root = '../dataset/'
path_train = root + Dataset + '/train_data.json'
path_save = root + Dataset + '/hypergraph_embeddings.json'
print('Reading data...')
with open(path_train) as f:
    line = f.readline()
    data = json.loads(line)
f.close()
user_number = len(data)
item_number = 0
for item_list in data: item_number = max(item_number, max(item_list))
item_number += 1

print('Initializing...')
H_u = sp.sparse.lil_matrix((user_number, item_number))
H_i = sp.sparse.lil_matrix((item_number, user_number))
I_u = sp.sparse.lil_matrix((user_number, user_number))
for u in range(user_number): I_u[u, u] = 1
I_i = sp.sparse.lil_matrix((item_number, item_number))
for i in range(item_number): I_i[i, i] = 1
W_u = np.zeros(user_number)
W_i = np.zeros(item_number)

# constructing the laplacian matrices
print('Constructing the laplacian matrices...')
for u in range(user_number):
    for i in data[u]:
        H_u[u, i] = 1
        H_i[i, u] = 1
        W_u[u] += 1
        W_i[i] += 1
if IF_WEIGHTED:
    for u in range(user_number):
        W_u[u] = 1.0 / max(W_u[u], epsilon)
    for i in range(item_number):
        W_i[i] = 1.0 / max(W_i[i], epsilon)
else: W_u, W_i = np.ones(user_number), np.ones(item_number)

print('   constructing user matrix...')
D_u = np.zeros(user_number)
D_i = np.zeros(item_number)
for u in range(user_number):
    for i in data[u]:
        D_u[u] += W_i[i]
        D_i[i] += 1
D_n = sp.sparse.lil_matrix((user_number, user_number))
D_e = sp.sparse.lil_matrix((item_number, item_number))
W_e = sp.sparse.lil_matrix((item_number, item_number))
for u in range(user_number):
    D_n[u, u] = 1.0 / max(sqrt(D_u[u]), epsilon)
for i in range(item_number):
    D_e[i, i] = 1.0 / max(D_i[i], epsilon)
    W_e[i, i] = W_i[i]
L_u = I_u - D_n * H_u * W_e * D_e * H_i * D_n

print('   constructing item matrix...')
D_u = np.zeros(user_number)
D_i = np.zeros(item_number)
for u in range(user_number):
    for i in data[u]:
        D_u[u] += 1
        D_i[i] += W_u[u]
D_n = sp.sparse.lil_matrix((item_number, item_number))
D_e = sp.sparse.lil_matrix((user_number, user_number))
W_e = sp.sparse.lil_matrix((user_number, user_number))
for i in range(item_number):
    D_n[i, i] = 1.0 / max(sqrt(D_i[i]), epsilon)
for u in range(user_number):
    D_e[u, u] = 1.0 / max(D_u[u], epsilon)
    W_e[u, u] = W_u[u]
L_i = I_i - D_n * H_i * W_e * D_e * H_u * D_n

#eigenvalue factorization
print('Decomposing the laplacian matrices...')
print('   decomposing user matrix...')
[Lamda, user_hypergraph_embeddings] = sp.sparse.linalg.eigsh(L_u, k = FREQUENCY_U, which='SM', tol = tolerant)
print(Lamda[0:10])
print('   decomposing item matrix...')
[Lamda, item_hypergraph_embeddings] = sp.sparse.linalg.eigsh(L_i, k = FREQUENCY_I, which='SM', tol = tolerant)
print(Lamda[0:10])

print('Saving features...')
f = open(path_save, 'w')
jsObj = json.dumps([user_hypergraph_embeddings.tolist(), item_hypergraph_embeddings.tolist()])
f.write(jsObj)
f.close()

