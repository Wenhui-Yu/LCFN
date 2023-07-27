## hyper-parameter setting
## author@Wenhui Yu  2023.07.09
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.02, "BPR": 0.1, "WBPR": 0.0001, "ShiftMC": 0.05, "DLNRS": 0.002},
           "Movielens": {"CrossEntropy": 0.01, "BPR": 0.02, "WBPR": 0.00002, "ShiftMC": 0.02, "DLNRS": 0.001},
           "KuaiRand": {"CrossEntropy": 0.02, "BPR": 0.05, "WBPR": 0.00002, "ShiftMC": 0.02, "DLNRS": 0.001}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.05, "BPR": 0.02, "WBPR": 0.01, "ShiftMC": 0.05, "DLNRS": 0.05},
              "Movielens": {"CrossEntropy": 0.02, "BPR": 0.01, "WBPR": 0.00005, "ShiftMC": 0.02, "DLNRS": 0.01},
              "KuaiRand": {"CrossEntropy": 0.01, "BPR": 0.005, "WBPR": 0.00005, "ShiftMC": 0.02, "DLNRS": 0.01}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][0]

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE, 'LOSS_FUNCTION': LOSS_FUNCTION,
            'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER, 'AUX_LOSS_WEIGHT': AUX_LOSS_WEIGHT, 'RHO': RHO}
