## hyper-parameter setting
## author@Wenhui Yu  2021.02.16
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.0002, "BPR": 0.0002, "WBPR": 0.00001, "ShiftMC": 0.01, "DLNRS": 0.0001},
           "KuaiRand": {"CrossEntropy": 0.0002, "BPR": 0.0005, "WBPR": 0.0005, "ShiftMC": 0.002, "DLNRS": 0.0005}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.01, "BPR": 0.5, "WBPR": 0.0002, "ShiftMC": 0.2, "DLNRS": 0.02},
              "KuaiRand": {"CrossEntropy": 0.001, "BPR": 0.02, "WBPR": 0.0005, "ShiftMC": 0.2, "DLNRS": 0.005}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][3]
LAYER = 2

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE, 'LAYER': LAYER,
            'LOSS_FUNCTION': LOSS_FUNCTION, 'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER, 'AUX_LOSS_WEIGHT': AUX_LOSS_WEIGHT,
            'RHO': RHO}
