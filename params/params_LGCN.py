## hyper-parameter setting
## author@Wenhui Yu  2021.02.16
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.0005, "BPR": 0.0005, "WBPR": 0.0005, "ShiftMC": 0.01, "DLNRS": 0.0005},
           "Movielens": {"CrossEntropy": 0.0005, "BPR": 0.0005, "WBPR": 0.0005, "ShiftMC": 0.002, "DLNRS": 0.0005},
           "KuaiRand": {"CrossEntropy": 0.001, "BPR": 0.0005, "WBPR": 0.0005, "ShiftMC": 0.002, "DLNRS": 0.0005}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.1, "BPR": 0.05, "WBPR": 2, "ShiftMC": 0.5, "DLNRS": 0.05},
              "Movielens": {"CrossEntropy": 0.05, "BPR": 0.02, "WBPR": 10, "ShiftMC": 0.1, "DLNRS": 0.01},
              "KuaiRand": {"CrossEntropy": 0.005, "BPR": 0.01, "WBPR": 10, "ShiftMC": 0.2, "DLNRS": 0.0002}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
SAMPLER = {"Amazon": 'LightGCN', "Movielens": 'LightGCN', "KuaiRand": 'LightGCN'}[DATASET]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][3]
FREQUENCY = 128
LAYER = 1

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE, 'LAYER': LAYER,
            'FREQUENCY': FREQUENCY, 'LOSS_FUNCTION': LOSS_FUNCTION, 'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER,
            'AUX_LOSS_WEIGHT': AUX_LOSS_WEIGHT, 'RHO': RHO}
