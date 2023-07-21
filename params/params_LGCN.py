## hyper-parameter setting
## author@Wenhui Yu  2021.02.16
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.0005, "BPR": 0.01, "WBPR": 0.0005, "ShiftMC": 0.01, "DLNRS": 0.0005},
           "KuaiRand": {"CrossEntropy": 0.001, "BPR": 0.01, "WBPR": 0.0005, "ShiftMC": 0.01, "DLNRS": 0.0005}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.1, "BPR": 0.01, "WBPR": 2, "ShiftMC": 0.5, "DLNRS": 0.05},
              "KuaiRand": {"CrossEntropy": 0.005, "BPR": 0.01, "WBPR": 10, "ShiftMC": 0.01, "DLNRS": 0.0002}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][3]
FREQUENCY = {'Amazon': 128, 'KuaiRand': 128}[DATASET]
LAYER = 1

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE, 'LAYER': LAYER,
            'FREQUENCY': FREQUENCY, 'LOSS_FUNCTION': LOSS_FUNCTION, 'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER,
            'AUX_LOSS_WEIGHT': AUX_LOSS_WEIGHT, 'RHO': RHO}
