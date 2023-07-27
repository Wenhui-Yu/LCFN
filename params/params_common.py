## hyper-parameter setting
## author@Wenhui Yu  2023.07.09
## email: jianlin.ywh@alibaba-inc.com

GPU_INDEX = "0"
DATASET = ['Amazon', 'Movielens', 'KuaiRand'][0]
MODEL = ['MF', 'NCF', 'NGCF', 'LightGCN', 'LGCN'][0]
LOSS_FUNCTION = ['CrossEntropy', 'BPR', 'WBPR', 'ShiftMC', 'DLNRS'][0]
EMB_DIM = 128
BATCH_SIZE = 10000
TEST_USER_BATCH = 4096
N_EPOCH = 200
IF_PRETRAIN = [False, True][0]
TEST_VALIDATION = 'Validation'  # can be changed automatically
TOP_K = [10, 20, 50, 100]
SAMPLE_RATE = 1
AUX_LOSS_WEIGHT = 0
RHO = 0.1
