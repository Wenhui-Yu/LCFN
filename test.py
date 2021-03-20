from train_model import train_model
from print_save import print_params, save_params
import tensorflow as tf
import random as rd
import time

def test(path_excel_dir, para_name, para, data, iter_num, vali_or_test):
    para[11] = vali_or_test
    for i in range(iter_num):
        print_params(para_name, para)
        path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
        save_params(para_name, para, path_excel)
        _ = train_model(para, data, path_excel)
        if para[2] not in ['GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN']: tf.reset_default_graph()
