## hyper-parameter setting
## author@Wenhui Yu  2021.02.16
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.002, "BPR": 0.002, "WBPR": 0.0001, "ShiftMC": 0.002, "DLNRS": 0.002},
           "Movielens": {"CrossEntropy": 0.001, "BPR": 0.002, "WBPR": 0.001, "ShiftMC": 0.005, "DLNRS": 0.002}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.1, "BPR": 0.05, "WBPR": 0.002, "ShiftMC": 0.1, "DLNRS": 0.05},
              "Movielens": {"CrossEntropy": 0.005, "BPR": 0.002, "WBPR": 5, "ShiftMC": 0.05, "DLNRS": 0.02}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][3]
LAYER = 2

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE, 'LAYER': LAYER,
            'LOSS_FUNCTION': LOSS_FUNCTION, 'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER, 'AUX_LOSS_WEIGHT':AUX_LOSS_WEIGHT,
            'RHO': RHO}
