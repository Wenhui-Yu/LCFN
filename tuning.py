## author@ Wenhui Yu  email: jianlin.ywh@alibaba-inc.com  2021.02.16
## Low-pass Collaborative Filtering Network (LCFN) and baselines

from train_model import train_model
from print_save import print_params, save_params
from get_hyperparameters import get_hyperparameter
import tensorflow as tf
import numpy as np
import random as rd
import time

def tuning(path_excel_dir, para_name, para, data, lr, lamda, min_num_coarse, max_num_coarse, min_num_fine, max_num_fine):
    ## tuning settings
    x_cen, y_cen = 1, 1
    score_matrix = np.zeros((3,3))
    hyper_matrix = np.array([[[lr * (10 ** i), lamda * (10 ** j)] for i in range(-1, 2)] for j in range(-1, 2)])
    num_matrix = np.zeros((3,3))
    ## coarse tuning
    for i in range(2):
        for x_curr, y_curr in [[x_cen, y_cen], [x_cen - 1, y_cen], [x_cen, y_cen - 1], [x_cen + 1, y_cen], [x_cen, y_cen + 1]]:
            if (num_matrix[x_curr, y_curr] < min_num_coarse or (x_curr == x_cen and y_curr == y_cen)) and (num_matrix[x_curr, y_curr] < 0.5 or score_matrix[x_curr, y_curr] >= 0.7 * score_matrix.max()):
                para[3: 5] = hyper_matrix[x_curr, y_curr]
                print_params(para_name, para)
                path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
                save_params(para_name, para, path_excel)
                score = train_model(para, data, path_excel)
                if para[2] not in ['GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN']: tf.reset_default_graph()
                score_matrix[x_curr, y_curr] = (score_matrix[x_curr, y_curr] * num_matrix[x_curr, y_curr] + score)/(num_matrix[x_curr, y_curr] + 1)
                num_matrix[x_curr, y_curr] += 1
                print(score_matrix)
                print(num_matrix)
                x_argmax, y_argmax = np.where(score_matrix == score_matrix.max())
                x_argmax, y_argmax = x_argmax[0], y_argmax[0]
                print('When \eta and \lambda is: ', hyper_matrix[x_argmax, y_argmax])
                print('the model achieves the best performance: ', score_matrix.max())
    while num_matrix[x_cen, y_cen] < max_num_coarse or score_matrix.max() != score_matrix[x_cen, y_cen]:
        x_cen, y_cen = np.where(score_matrix == score_matrix.max())
        x_cen, y_cen = x_cen[0], y_cen[0]
        ## extending the matrices
        if y_cen == 0:
            y_cen += 1
            pad = np.zeros(score_matrix.shape[0])
            score_matrix = np.c_[pad, score_matrix]
            num_matrix = np.c_[pad, num_matrix]
            hyper_matrix = np.concatenate((np.zeros((hyper_matrix.shape[0], 1, 2)), hyper_matrix), axis=1)
            for i in range(hyper_matrix.shape[0]):
                hyper_matrix[i, 0, 0] = hyper_matrix[i, 1, 0] / 10
                hyper_matrix[i, 0, 1] = hyper_matrix[i, 1, 1]
        if y_cen == score_matrix.shape[1] - 1:
            pad = np.zeros(score_matrix.shape[0])
            score_matrix = np.c_[score_matrix, pad]
            num_matrix = np.c_[num_matrix, pad]
            hyper_matrix = np.concatenate((hyper_matrix, np.zeros((hyper_matrix.shape[0], 1, 2))), axis=1)
            for i in range(hyper_matrix.shape[0]):
                hyper_matrix[i, -1, 0] = hyper_matrix[i, -1-1, 0] * 10
                hyper_matrix[i, -1, 1] = hyper_matrix[i, -1-1, 1]
        if x_cen == 0:
            x_cen += 1
            pad = np.zeros((1, score_matrix.shape[1]))
            score_matrix = np.r_[pad, score_matrix]
            num_matrix = np.r_[pad, num_matrix]
            hyper_matrix = np.concatenate((np.zeros((1, hyper_matrix.shape[1], 2)), hyper_matrix), axis=0)
            for i in range(hyper_matrix.shape[1]):
                hyper_matrix[0, i, 0] = hyper_matrix[1, i, 0]
                hyper_matrix[0, i, 1] = hyper_matrix[1, i, 1] / 10
        if x_cen == score_matrix.shape[0] - 1:
            pad = np.zeros((1, score_matrix.shape[1]))
            score_matrix = np.r_[score_matrix, pad]
            num_matrix = np.r_[num_matrix, pad]
            hyper_matrix = np.concatenate((hyper_matrix, np.zeros((1, hyper_matrix.shape[1], 2))), axis=0)
            for i in range(hyper_matrix.shape[1]):
                hyper_matrix[-1, i, 0] = hyper_matrix[-1-1, i, 0]
                hyper_matrix[-1, i, 1] = hyper_matrix[-1-1, i, 1] * 10
        ## finding the best performance
        for x_curr, y_curr in [[x_cen, y_cen], [x_cen - 1, y_cen], [x_cen, y_cen - 1], [x_cen + 1, y_cen], [x_cen, y_cen + 1]]:
            if (num_matrix[x_curr, y_curr] < min_num_coarse or (x_curr == x_cen and y_curr == y_cen)) and (num_matrix[x_curr, y_curr] < 0.5 or score_matrix[x_curr, y_curr] >= 0.7 * score_matrix.max()):
                para[3: 5] = hyper_matrix[x_curr, y_curr]
                print_params(para_name, para)
                path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
                save_params(para_name, para, path_excel)
                score = train_model(para, data, path_excel)
                if para[2] not in ['GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN']: tf.reset_default_graph()
                score_matrix[x_curr, y_curr] = (score_matrix[x_curr, y_curr] * num_matrix[x_curr, y_curr] + score)/(num_matrix[x_curr, y_curr] + 1)
                num_matrix[x_curr, y_curr] += 1
                print(score_matrix)
                print(num_matrix)
                x_argmax, y_argmax = np.where(score_matrix == score_matrix.max())
                x_argmax, y_argmax = x_argmax[0], y_argmax[0]
                print('When \eta and \lambda is: ', hyper_matrix[x_argmax, y_argmax])
                print('the model achieves the best performance: ', score_matrix.max())
    ## fine tuning
    x_cen, y_cen = np.where(score_matrix == score_matrix.max())
    x_cen, y_cen = x_cen[0], y_cen[0]
    score_max = score_matrix[x_cen, y_cen]
    num_max = num_matrix[x_cen, y_cen]
    lr, lamda = hyper_matrix[x_cen, y_cen]
    x_cen, y_cen = 2, 2
    score_matrix = np.zeros((5, 5))
    score_matrix[x_cen, y_cen] = score_max
    num_matrix = np.zeros((5, 5))
    num_matrix[x_cen, y_cen] = num_max
    hyper_matrix_lr = np.expand_dims(np.array([get_hyperparameter(lr) for i in range(5)]), axis=-1)
    hyper_matrix_lamda = np.expand_dims(np.array([get_hyperparameter(lamda) for i in range(5)]).T, axis=-1)
    hyper_matrix = np.concatenate((hyper_matrix_lr, hyper_matrix_lamda), axis=-1)
    ## initializing the score matrix
    for i in range(3):
        for x_curr, y_curr in [[1,1],[1,3],[3,1],[3,3]]:
            para[3: 5] = hyper_matrix[x_curr, y_curr]
            print_params(para_name, para)
            path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
            save_params(para_name, para, path_excel)
            score = train_model(para, data, path_excel)
            if para[2] not in ['GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN']: tf.reset_default_graph()
            score_matrix[x_curr, y_curr] = (score_matrix[x_curr, y_curr] * num_matrix[x_curr, y_curr] + score) / (num_matrix[x_curr, y_curr] + 1)
            num_matrix[x_curr, y_curr] += 1
            print(score_matrix)
            print(num_matrix)
            x_argmax, y_argmax = np.where(score_matrix == score_matrix.max())
            x_argmax, y_argmax = x_argmax[0], y_argmax[0]
            print('When \eta and \lambda is: ', hyper_matrix[x_argmax, y_argmax])
            print('the model achieves the best performance: ', score_matrix.max())
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
                para[3: 5] = hyper_matrix[x_curr, y_curr]
                print_params(para_name, para)
                path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
                save_params(para_name, para, path_excel)
                score = train_model(para, data, path_excel)
                if para[2] not in ['GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN']: tf.reset_default_graph()
                score_matrix[x_curr, y_curr] = (score_matrix[x_curr, y_curr] * num_matrix[x_curr, y_curr] + score)/(num_matrix[x_curr, y_curr] + 1)
                num_matrix[x_curr, y_curr] += 1
                print(score_matrix)
                print(num_matrix)
                x_argmax, y_argmax = np.where(score_matrix == score_matrix.max())
                x_argmax, y_argmax = x_argmax[0], y_argmax[0]
                print('When \eta and \lambda is: ', hyper_matrix[x_argmax, y_argmax])
                print('the model achieves the best performance: ', score_matrix.max())
