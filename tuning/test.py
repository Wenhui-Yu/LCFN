from utils.train_model import train_model
from utils.print_save import print_params, save_params
from utils.dense2sparse import propagation_matrix
import tensorflow as tf
import random as rd
import time

def test(path_excel_dir, para, data, iter_num):
    for i in range(iter_num):
        print_params(para)
        path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
        save_params(para, path_excel)
        _ = train_model(para, data, path_excel)
        tf.reset_default_graph()
        if para['MODEL'] in ['NGCF', 'LightGCN'] or para['SAMPLER'] in ['NGCF', 'LightGCN']:
            data[-1] = propagation_matrix(data[1], data[3], data[4], 'sym_norm')