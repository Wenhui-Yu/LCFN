## author@ Wenhui Yu  email:yuwh16@tsinghua.edu.cn  2019.04.14
## Low-pass Collaborative Filtering Network (LCFN) and baselines

from train_model import *
from params import *
from print_save import *
import time
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if __name__ == '__main__':
    for ii in range(1):
        for iLR in [0.00001,0.0001,0.001]:
            for iLAMDA in [0.0001,0.001,0.01]:
#                 LR = 0.02
#                 LAYER = 3
#                 EMB_DIM = 32
                path_excel = 'experiment_result/'+DATASET+'_'+MODEL+'_'+str(int(time.time()))+str(int(random.uniform(100,900)))+'.xlsx'
                para = [DATASET,MODEL,LR,LAMDA,LAYER,EMB_DIM,FREQUENCY_USER,FREQUENCY_ITEM,FREQUENCY,KEEP_RATE,
                        BATCH_SIZE,SAMPLE_RATE,IF_PRETRAIN,N_EPOCH,TEST_VALIDATION,TOP_K,OPTIMIZATION]
                para_name = ['DATASET','MODEL','LR','LAMDA','LAYER','EMB_DIM','FREQUENCY_USER','FREQUENCY_ITEM','FREQUENCY','KEEP_RATE',
                             'BATCH_SIZE','SAMPLE_RATE','IF_PRETRAIN','N_EPOCH','TEST_VALIDATION','TOP_K','OPTIMIZATION']
                ## print and save model hyperparameters
                print('GPU INDEX: ', GPU_INDEX)
                print_params(para_name, para)
                save_params(para_name, para, path_excel)
                ## train the model
                train_model(para, path_excel)
                