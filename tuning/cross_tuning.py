from utils.train_model import train_model
from utils.print_save import print_params, save_params
from utils.dense2sparse import propagation_matrix
from params.get_hyperparameters import get_hyperparameter
import tensorflow as tf
import numpy as np
import random as rd
import time

def cross_tuning(path_excel_dir, para, data, lr, lamda, min_num_fine, max_num_fine):
    ## fine tuning
    x_cen, y_cen = 2, 2
    score_matrix = np.zeros((5, 5))
    score_matrix[x_cen, y_cen] = 0.1 ** 10
    num_matrix = np.zeros((5, 5))
    hyper_matrix_lr = np.expand_dims(np.array([get_hyperparameter(lr) for i in range(5)]), axis=-1)
    hyper_matrix_lamda = np.expand_dims(np.array([get_hyperparameter(lamda) for i in range(5)]).T, axis=-1)
    hyper_matrix = np.concatenate((hyper_matrix_lr, hyper_matrix_lamda), axis=-1)
    while num_matrix[x_cen, y_cen] < max_num_fine or score_matrix.max() != score_matrix[x_cen, y_cen]:
        x_cen, y_cen = np.where(score_matrix == score_matrix.max())
        x_cen, y_cen = x_cen[0], y_cen[0]
        ## extending matrices
        if y_cen == 0:
            y_cen += 1
            pad = np.zeros(score_matrix.shape[0])
            score_matrix = np.c_[pad, score_matrix]
            num_matrix = np.c_[pad, num_matrix]
            pad_lr = np.ones(hyper_matrix.shape[0]) * get_hyperparameter(hyper_matrix[x_cen, y_cen, 0])[1]
            pad_lamda = hyper_matrix[:, 0, 1]
            hyper_matrix = np.concatenate((np.zeros((hyper_matrix.shape[0], 1, 2)), hyper_matrix), axis=1)
            hyper_matrix[:, 0, 0] = pad_lr
            hyper_matrix[:, 0, 1] = pad_lamda
        if y_cen == score_matrix.shape[1] - 1:
            pad = np.zeros(score_matrix.shape[0])
            score_matrix = np.c_[score_matrix, pad]
            num_matrix = np.c_[num_matrix, pad]
            pad_lr = np.ones(hyper_matrix.shape[0]) * get_hyperparameter(hyper_matrix[x_cen, y_cen, 0])[3]
            pad_lamda = hyper_matrix[:, 0, 1]
            hyper_matrix = np.concatenate((hyper_matrix, np.zeros((hyper_matrix.shape[0], 1, 2))), axis=1)
            hyper_matrix[:, -1, 0] = pad_lr
            hyper_matrix[:, -1, 1] = pad_lamda
        if x_cen == 0:
            x_cen += 1
            pad = np.zeros((1, score_matrix.shape[1]))
            score_matrix = np.r_[pad, score_matrix]
            num_matrix = np.r_[pad, num_matrix]
            pad_lr = hyper_matrix[0, :, 0]
            pad_lamda = np.ones(hyper_matrix.shape[1]) * get_hyperparameter(hyper_matrix[x_cen, y_cen, 1])[1]
            hyper_matrix = np.concatenate((np.zeros((1, hyper_matrix.shape[1], 2)), hyper_matrix), axis=0)
            hyper_matrix[0, :, 0] = pad_lr
            hyper_matrix[0, :, 1] = pad_lamda
        if x_cen == score_matrix.shape[0] - 1:
            pad = np.zeros((1, score_matrix.shape[1]))
            score_matrix = np.r_[score_matrix, pad]
            num_matrix = np.r_[num_matrix, pad]
            pad_lr = hyper_matrix[0, :, 0]
            pad_lamda = np.ones(hyper_matrix.shape[1]) * get_hyperparameter(hyper_matrix[x_cen, y_cen, 1])[3]
            hyper_matrix = np.concatenate((hyper_matrix, np.zeros((1, hyper_matrix.shape[1], 2))), axis=0)
            hyper_matrix[-1, :, 0] = pad_lr
            hyper_matrix[-1, :, 1] = pad_lamda
        ## finding the best performance
        for x_curr, y_curr in [[x_cen, y_cen], [x_cen - 1, y_cen], [x_cen, y_cen - 1], [x_cen + 1, y_cen], [x_cen, y_cen + 1]]:
            if (num_matrix[x_curr, y_curr] < min_num_fine or (x_curr == x_cen and y_curr == y_cen)) and (num_matrix[x_curr, y_curr] < 0.5 or score_matrix[x_curr, y_curr] >= 0.7 * score_matrix.max()):
                para["LR"], para["LAMDA"] = hyper_matrix[x_curr, y_curr]
                print_params(para)
                path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
                save_params(para, path_excel)
                score = train_model(para, data, path_excel)
                score_matrix[x_curr, y_curr] = (score_matrix[x_curr, y_curr] * num_matrix[x_curr, y_curr] + score)/(num_matrix[x_curr, y_curr] + 1)
                num_matrix[x_curr, y_curr] += 1
                print(score_matrix)
                print(num_matrix)
                x_argmax, y_argmax = np.where(score_matrix == score_matrix.max())
                x_argmax, y_argmax = x_argmax[0], y_argmax[0]
                print('When \eta and \lambda is: ', hyper_matrix[x_argmax, y_argmax])
                print('the model achieves the best performance: ', score_matrix.max())
                tf.reset_default_graph()
                if para['MODEL'] in ['NGCF', 'LightGCN'] or para['SAMPLER'] in ['NGCF', 'LightGCN']:
                    data[-1] = propagation_matrix(data[1], data[3], data[4], 'sym_norm')
