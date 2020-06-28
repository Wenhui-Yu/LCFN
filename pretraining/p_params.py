## author@Wenhui Yu  2020.06.02
## email: yuwh16@mails.tsinghua.edu.cn

dataset = 0         # 0:Amazon, 1:Movielens
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = 'MF_BPR'
LR = [0.05, 0.02][dataset]
LAMDA = [0.02, 0.01][dataset]
EMB_DIM = 64
BATCH_SIZE = 10000
SAMPLE_RATE = 1
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [10]
DIR = '../dataset/'+DATASET+'/'
import os