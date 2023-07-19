## author@ Wenhui Yu  email: jianlin.ywh@alibaba-inc.com  2021.02.16

from params.params_common import MODEL
if MODEL == "MF": from params.params_MF import all_para
if MODEL == "NCF": from params.params_NCF import all_para
if MODEL == "NGCF": from params.params_NGCF import all_para
if MODEL == "LightGCN": from params.params_LightGCN import all_para
if MODEL == "LGCN": from params.params_LGCN import all_para
from tuning.tuning import tuning
from tuning.fine_tuning import fine_tuning
from tuning.cross_tuning import cross_tuning
from tuning.coarse_tuning import coarse_tuning
from tuning.test import test
from utils.read_data import read_all_data
import os

if __name__ == '__main__':
    tuning_method = ['tuning', 'fine_tuning', 'cross_tuning', 'coarse_tuning', 'test'][0]  # set here to tune model or test model
    lr_coarse, lamda_coarse = 0.001, 0.01       # start coarse search at
    lr_fine, lamda_fine = 0.0005, 0.1           # start fine search at, needed in only ``fine_tuning''
    min_num_coarse, max_num_coarse = 3, 5       # repeat numbers
    min_num_fine, max_num_fine = 10, 20
    iter_num_test = 20
    if tuning_method == 'test': all_para['TEST_VALIDATION'] = 'Test'    # if testing the model, we need to read in test set
    data = list(read_all_data(all_para))                                # read data
    path_excel_dir = 'experiment_result/' + all_para['DATASET'] + '_' + all_para['MODEL'] + '_'
    if all_para['LOSS_FUNCTION'] == 'DLNRS': path_excel_dir += all_para['SAMPLER'] + '_'

    ## tuning the model
    os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']
    if tuning_method == 'tuning': tuning(path_excel_dir, all_para, data, lr_coarse, lamda_coarse, min_num_coarse, max_num_coarse, min_num_fine, max_num_fine)
    if tuning_method == 'fine_tuning': fine_tuning(path_excel_dir, all_para, data, lr_fine, lamda_fine, min_num_fine, max_num_fine)
    if tuning_method == 'cross_tuning': cross_tuning(path_excel_dir, all_para, data, lr_fine, lamda_fine, min_num_fine, max_num_fine)
    if tuning_method == 'coarse_tuning': coarse_tuning(path_excel_dir, all_para, data, lr_coarse, lamda_coarse, min_num_coarse, max_num_coarse)
    if tuning_method == 'test': test(path_excel_dir, all_para, data, iter_num_test)
