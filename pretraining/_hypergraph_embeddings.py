## Hypergraph embeddings as the Fourier transform bases for LCFN model
## author@Wenhui Yu  2020.06.02
## email: yuwh16@mails.tsinghua.edu.cn

import os
import scipy as sp
import scipy.sparse.linalg
from numpy import *
import numpy as np
import json

dataset = 0     #0 for amazon, 1 for movielens
K_u = 20000
K_v = 10000
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10
Dataset = ['Amazon', 'Movielens'][dataset]
root = os.path.abspath(os.path.dirname(os.getcwd())) + '/dataset/'
path_train = root + Dataset + '/train_data.json'
path_save = root + Dataset + '/hypergraph_embeddings.json'

with open(path_train) as f:
    line = f.readline()
    data = json.loads(line)
f.close()
user_number = len(data)
item_number = 0
for i in data:
    item_number = max(item_number, max(i))
item_number += 1
#print(user_number, item_number)

H_u = sp.sparse.lil_matrix((user_number, item_number))
H_v = sp.sparse.lil_matrix((item_number, user_number))
D_u = sp.sparse.lil_matrix((user_number, user_number))
D_v = sp.sparse.lil_matrix((item_number, item_number))
I_u = sp.sparse.lil_matrix(np.eye(user_number, user_number))
I_v = sp.sparse.lil_matrix(np.eye(item_number, item_number))

# constructing the laplacian matrices
print('Constructing the laplacian matrices...')
for user in range(user_number):
    for item in data[user]:
        H_u[user, item] = 1
        H_v[item, user] = 1
        D_u[user, user] += 1
        D_v[item, item] += 1
# initialization
print('   constructing user matrix...')
D_n = sp.sparse.lil_matrix((user_number, user_number))
D_e = sp.sparse.lil_matrix((item_number, item_number))
for i in range(user_number):
    D_n[i, i] = 1.0 / max(sqrt(D_u[i, i]), epsilon)
for i in range(item_number):
    D_e[i, i] = 1.0 / max(D_v[i, i], epsilon)
L_u = I_u - D_n * H_u * D_e * H_u.T * D_n

print('   constructing item matrix...')
D_n = sp.sparse.lil_matrix((item_number, item_number))
D_e = sp.sparse.lil_matrix((user_number, user_number))
for i in range(item_number):
    D_n[i, i] = 1.0 / max(sqrt(D_v[i, i]), epsilon)
for i in range(user_number):
    D_e[i, i] = 1.0 / max(D_u[i, i], epsilon)
L_v = I_v - D_n * H_v * D_e * H_v.T * D_n

#eigenvalue factorization
print('Decomposing the laplacian matrices...')
print('   decomposing user matrix...')
[Lamda, user_graph_embeddings] = sp.sparse.linalg.eigsh(L_u, k = K_u, which='SM', tol = tolerant)
print(Lamda[0:10])
print('   decomposing item matrix...')
[Lamda, item_graph_embeddings] = sp.sparse.linalg.eigsh(L_v, k = K_v, which='SM', tol = tolerant)
print(Lamda[0:10])
print('Saving features...')

f = open(path_save, 'w')
jsObj = json.dumps([user_graph_embeddings.tolist(), item_graph_embeddings.tolist()])
f.write(jsObj)
f.close()


'''
import scipy as sp
import scipy.sparse.linalg
from numpy import *
import numpy as np
import json

K_u = 1400
K_v = 1300
tolerant = 0.1 ** 5
d = 0.1 ** 10
path_train = 'train_data.json'
path_save = 'hypergraph_embeddings.json'

user_number = 0
item_number = 0
with open(path_train) as f:
    line = f.readline()
    data = json.loads(line)
f.close()
user_number = len(data)
for i in data:
    item_number = max(item_number, max(i))
item_number += 1
#print(user_number, item_number)

H_u = sp.sparse.lil_matrix((user_number, item_number))
H_v = sp.sparse.lil_matrix((item_number, user_number))
D_u = sp.sparse.lil_matrix((user_number, user_number))
D_v = sp.sparse.lil_matrix((item_number, item_number))
I_u = sp.sparse.lil_matrix(np.eye(user_number, user_number))
I_v = sp.sparse.lil_matrix(np.eye(item_number, item_number))

# constructing the laplacian matrices
print('Constructing the laplacian matrices...')
for user in range(user_number):
    for item in data[user]:
        H_u[user, item] = 1
        H_v[item, user] = 1
        D_u[user, user] += 1
        D_v[item, item] += 1
# initialization
print('   constructing user matrix...')
D_n = sp.sparse.lil_matrix((user_number, user_number))
D_e = sp.sparse.lil_matrix((item_number, item_number))
for i in range(user_number):
    D_n[i, i] = 1.0 / max(sqrt(D_u[i, i]), epsilon)
for i in range(item_number):
    D_e[i, i] = 1.0 / max(D_v[i, i], epsilon)
L_u = I_u - D_n * H_u * D_e * H_u.T * D_n

print('   constructing item matrix...')
D_n = sp.sparse.lil_matrix((item_number, item_number))
D_e = sp.sparse.lil_matrix((user_number, user_number))
for i in range(item_number):
    D_n[i, i] = 1.0 / max(sqrt(D_v[i, i]), epsilon)
for i in range(user_number):
    D_e[i, i] = 1.0 / max(D_u[i, i], epsilon)
L_v = I_v - D_n * H_v * D_e * H_v.T * D_n

#eigenvalue factorization
print('Decomposing the laplacian matrices...')
print('   decomposing user matrix...')
[Lamda, user_graph_embeddings] = sp.sparse.linalg.eigsh(L_u, k = K_u, which='SM', tol = tolerant)
print(Lamda[0:10])
print('   decomposing item matrix...')
[Lamda, item_graph_embeddings] = sp.sparse.linalg.eigsh(L_v, k = K_v, which='SM', tol = tolerant)
print(Lamda[0:10])
print('Saving features...')

result = np.zeros((user_number+item_number, max(K_u, K_v)))
result[0: user_number, 0: K_u] = user_graph_embeddings
result[user_number: user_number+item_number, 0: K_v] = item_graph_embeddings
f = open(path_save, 'w')
jsObj = json.dumps(result.tolist())
f.write(jsObj)
f.close()
'''

'''
H_u = mat(np.zeros((user_number, item_number)))
H_v = mat(np.zeros((item_number, user_number)))
D_u = mat(np.zeros((user_number, user_number)))
D_v = mat(np.zeros((item_number, item_number)))
I_u = mat(np.eye(user_number, user_number))
I_v = mat(np.eye(item_number, item_number))

# constructing the laplacian matrices
print('Constructing the laplacian matrices...')
for user in range(user_number):
    for item in data[user]:
        H_u[user, item] = 1
        H_v[item, user] = 1
        D_u[user, user] += 1
        D_v[item, item] += 1
# initialization
print('   constructing user matrix...')
D_n = sp.sparse.lil_matrix((user_number, user_number))
D_e = sp.sparse.lil_matrix((item_number, item_number))
for i in range(user_number):
    D_n[i, i] = 1.0 / (sqrt(D_u[i, i]) + e)
for i in range(item_number):
    D_e[i, i] = 1.0 / (D_v[i, i] + e)
L_u = I_u - D_n * H_u * D_e * H_u.T * D_n

print('   constructing item matrix...')
D_n = sp.sparse.lil_matrix((item_number, item_number))
D_e = sp.sparse.lil_matrix((user_number, user_number))
for i in range(item_number):
    D_n[i, i] = 1.0 / (sqrt(D_v[i, i]) + e)
for i in range(user_number):
    D_e[i, i] = 1.0 / (D_u[i, i] + e)
L_v = I_v - D_n * H_v * D_e * H_v.T * D_n
L_u = np.array(L_u)
lamda, U = np.linalg.eig(L_u)
print(lamda.sort())

'''
