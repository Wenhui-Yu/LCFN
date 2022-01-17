## author@ Wenhui Yu  email: jianlin.ywh@alibaba-inc.com  2021.02.16
## run models: our models (LCFN, LGCN, and SGNN) and baselines
from params import all_para, pred_dim
from change_params import change_params
from tuning import tuning
from fine_tuning import fine_tuning
from cross_tuning import cross_tuning
from coarse_tuning import coarse_tuning
from test import test
from read_data import read_all_data
import os

if __name__ == '__main__':
    ## change hyperparameters temporarily here
    change_dic = {
        # 'ACTIVATION': ['None', 'Tanh', 'Sigmoid', 'ReLU'][0],
        # 'dataset': 1,   # 0:Amazon, 1:Movielens
        # 'model': 8,     # 0:MF, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:LightGCN, 7:LCFN, 8:LGCN, 9:SGNN
    }
    all_para = change_params(all_para, change_dic, pred_dim)

    ## setting tuning strategies here
    path_excel_dir = 'experiment_result/' + all_para[1] + '_' + all_para[2] + '_'
    tuning_method = ['tuning', 'fine_tuning', 'cross_tuning', 'coarse_tuning', 'test'][4]  ## set here to tune model or test model
    ## initial hyperparameter settings
    lr_coarse, lamda_coarse = 0.001, 0.01
    lr_fine, lamda_fine = 0.0005, 0.1
    ## repeat numbers
    min_num_coarse, max_num_coarse = 3, 5
    min_num_fine, max_num_fine = 10, 50
    iter_num_test = 20

    ## select hyperparameters for different model
    para = all_para[0: 13]
    if all_para[2] == 'LCFN': para += all_para[13: 15]
    if all_para[2] == 'LGCN': para += all_para[13: 26]
    if all_para[2] == 'SGNN': para += all_para[26: 29]
    para_name = ['GPU_INDEX', 'DATASET', 'MODEL', 'LR', 'LAMDA', 'LAYER', 'EMB_DIM', 'BATCH_SIZE', 'TEST_USER_BATCH', 'N_EPOCH', 'IF_PRETRAIN', 'TEST_VALIDATION', 'TOP_K']
    if all_para[2] == 'LCFN': para_name += ['FREQUENCY_USER', 'FREQUENCY_ITEM']
    if all_para[2] == 'LGCN': para_name += ['FREQUENCY_USER', 'FREQUENCY_ITEM', 'FREQUENCY', 'KEEP_PORB', 'SAMPLE_RATE', 'GRAPH_CONV', 'PREDICTION', 'LOSS_FUNCTION', 'GENERALIZATION', 'OPTIMIZATION', 'IF_TRASFORMATION', 'ACTIVATION', 'POOLING']
    if all_para[2] == 'SGNN': para_name += ['PROP_DIM', 'PROP_EMB', 'IF_NORM']
    # if testing the model, we need to read in test set
    if tuning_method == 'test': all_para[11] = para[11] = 'Test'

    ## read data
    data = read_all_data(all_para)
    para[10] = data[-1]

    ## tuning the model
    os.environ["CUDA_VISIBLE_DEVICES"] = all_para[0]
    if tuning_method == 'tuning': tuning(path_excel_dir, para_name, para, data, lr_coarse, lamda_coarse, min_num_coarse, max_num_coarse, min_num_fine, max_num_fine)
    if tuning_method == 'fine_tuning': fine_tuning(path_excel_dir, para_name, para, data, lr_fine, lamda_fine, min_num_fine, max_num_fine)
    if tuning_method == 'cross_tuning': cross_tuning(path_excel_dir, para_name, para, data, lr_fine, lamda_fine, min_num_fine, max_num_fine)
    if tuning_method == 'coarse_tuning': coarse_tuning(path_excel_dir, para_name, para, data, lr_coarse, lamda_coarse, min_num_coarse, max_num_coarse)
    if tuning_method == 'test': test(path_excel_dir, para_name, para, data, iter_num_test)
