## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

dataset = 0         # 0:Amazon, 1:KuaiRand
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Amazon', 'KuaiRand'][dataset]
MODEL = 'MF'
LR = [0.001, 0.02][dataset]
LAMDA = [0.2, 0.01][dataset]
EMB_DIM = 128
BATCH_SIZE = 10000
TEST_USER_BATCH = {'Amazon': 4096, 'KuaiRand': 4096}[DATASET]
SAMPLE_RATE = 1
N_EPOCH = 200
TOP_K = [10, 20, 50, 100]
DIR = '../dataset/'+DATASET+'/'
GPU_INDEX = "0"

all_para = {'GPU_INDEX': GPU_INDEX, 'DATASET': DATASET, 'MODEL': MODEL, 'LR': LR, 'LAMDA': LAMDA, 'EMB_DIM': EMB_DIM,
            'BATCH_SIZE': BATCH_SIZE, 'TEST_USER_BATCH': TEST_USER_BATCH, 'N_EPOCH': N_EPOCH, 'IF_PRETRAIN': False,
            'TEST_VALIDATION': 'Validation', 'TOP_K': TOP_K, 'SAMPLE_RATE': SAMPLE_RATE,'LOSS_FUNCTION': 'CrossEntropy',
            'OPTIMIZER': 'SGD', 'SAMPLER': 'MF', 'AUX_LOSS_WEIGHT': 0}