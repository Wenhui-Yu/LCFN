## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

dataset = 0         # 0:Amazon, 1:Movielens
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = 'MF'
LR = [0.05, 0.02][dataset]
LAMDA = [0.02, 0.01][dataset]
EMB_DIM = 128
BATCH_SIZE = 10000
TEST_USER_BATCH = [4096, 1024][dataset]
SAMPLE_RATE = 1
N_EPOCH = 200
TOP_K = [2, 5, 10, 20, 50, 100]
DIR = '../dataset/'+DATASET+'/'
GPU_INDEX = "0"