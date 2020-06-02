model = 1           # 0:MF_BPR, 1:MF_MSE
dataset = 0         # 0:Amazon, 1:Movielens
validate_test = 0   # 0:Validate, 1: Test

DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = ['MF_BPR', 'MF_MSE'][model]
LR = [[0.05,0.01], [0.02,0.02]][dataset][model]
LAMDA = [[0.02,0.02], [0.01,0.01]][dataset][model]
EMB_DIM = 128       #[[128,128], [128,64]][dataset][model]
BATCH_SIZE = 10000
SAMPLE_RATE = 1
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [2, 5, 10, 20, 50, 100]
DIR = '../dataset/'+DATASET+'/'
import os