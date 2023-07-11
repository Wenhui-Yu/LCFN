## hyper-parameter setting
## author@Wenhui Yu  2023.07.09
## email: jianlin.ywh@alibaba-inc.com

from params.params_common import *

LR_list = {"Amazon": {"CrossEntropy": 0.01, "BPR": 0.01, "WBPR": 0.01, "DLNRS": 0.01},
           "KuaiRand": {"CrossEntropy": 0.01, "BPR": 0.01, "WBPR": 0.01, "DLNRS": 0.01}}
LAMDA_list = {"Amazon": {"CrossEntropy": 0.01, "BPR": 0.01, "WBPR": 0.01, "DLNRS": 0.01},
              "KuaiRand": {"CrossEntropy": 0.01, "BPR": 0.01, "WBPR": 0.01, "DLNRS": 0.01}}
LR = LR_list[DATASET][LOSS_FUNCTION]
LAMDA = LAMDA_list[DATASET][LOSS_FUNCTION]
OPTIMIZER = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][0]

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': IF_PRETRAIN,
            'TEST_VALIDATION': TEST_VALIDATION, 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE,'LOSS_FUNCTION': LOSS_FUNCTION,
            'OPTIMIZER': OPTIMIZER, 'SAMPLER': SAMPLER}
