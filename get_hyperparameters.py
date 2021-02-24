## author@ Wenhui Yu email: jianlin.ywh@alibaba-inc.com  2021.02.16
## getting the hyperparameters

import numpy as np

def get_hyperparameter(x):
    para_list = np.array([0.00000001,0.00000002,0.00000005,
                          0.0000001,0.0000002,0.0000005,
                          0.000001,0.000002,0.000005,
                          0.00001,0.00002,0.00005,
                          0.0001,0.0002,0.0005,
                          0.001,0.002,0.005,
                          0.01,0.02,0.05,
                          0.1,0.2,0.5,
                          1,2,5,
                          10,20,50,
                          100,200,500])
    index = np.argwhere(para_list == x)
    return para_list[index[0][0]-2: index[0][0]+3].tolist()