## graph embeddings as the Fourier transform bases for LGCN model
## author@Wenhui Yu  2021.01.17
## email: jianlin.ywh@alibaba-inc.com

import scipy as sp
import scipy.sparse.linalg
from numpy import *
import json

DATASET = 0             # 0 for Amazon, 1 for KuaiRand
FREQUENCY = 128         # dimensionality of the base
Dataset = ['Amazon', 'KuaiRand'][DATASET]
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10

root = '../dataset/'
path_train = root + Dataset + '/train_data.json'
path_save = root + Dataset + '/graph_embeddings' + '.json'
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
