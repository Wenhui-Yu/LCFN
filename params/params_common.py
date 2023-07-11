## hyper-parameter setting
## author@Wenhui Yu  2023.07.09
## email: jianlin.ywh@alibaba-inc.com

GPU_INDEX = "0"
DATASET = ['Amazon', 'KuaiRand'][0]
MODEL = ['MF', 'NCF', 'SCF', 'LightGCN', 'LGCN'][0]
LOSS_FUNCTION = ['CrossEntropy', 'BPR', 'WBPR', 'DLNRS'][0]
EMB_DIM = 128
BATCH_SIZE = 10000
TEST_USER_BATCH = {'Amazon': 4096, 'KuaiRand': 4096}[DATASET]
N_EPOCH = 200
IF_PRETRAIN = [False, True][0]
TEST_VALIDATION = 'Validation'  # can be changed automatically
TOP_K = [10, 20, 50, 100]
SAMPLE_RATE = 1

