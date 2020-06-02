model = 0           # 0:BPR, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:LCFN, 7:SGNN
dataset = 0         # 0:Amazon, 1:Movielens
############################################################
validate_test = 0   # 0:Validate, 1: Test
DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = ['BPR', 'NCF', 'GCMC', 'NGCF', 'SCF', 'CGMC', 'LCFN', 'SGNN'][model]
OPTIMIZATION = ['SGD', 'SGD', 'Adam', 'Adam', 'RMSProp', 'Adam', 'Adam', 'Adam'][model]
LR = [[0.02,0.0002,0.001,0.0001,0.0001,0.0001,0.0005,0.0005], [0.02,0.00001,0.0002,0.00005,0.0001,0.00002,0.005,0.0002]][dataset][model]
LAMDA = [[0.02,0,0.05,0.001,0.02,0.0002,0.1,0.1], [0.01,0,0.02,0.02,0.01,0.05,0.05,0.02]][dataset][model]
LAYER = [[0,4,1,1,1,1,2,1], [0,4,1,1,1,1,3,1]][dataset][model]
EMB_DIM = [[128,64,64,64,64,64,43,64], [128,64,64,64,64,64,32,64]][dataset][model]
FREQUENCY_USER = [[0,0,0,0,0,0,4000,0], [0,0,0,0,0,0,3000,0]][dataset][model]
FREQUENCY_ITEM = [[0,0,0,0,0,0,2000,0], [0,0,0,0,0,0,2000,0]][dataset][model]
FREQUENCY = [[0,0,0,0,0,0,0,128], [0,0,0,0,0,0,0,128]][dataset][model]
KEEP_RATE = [[0.8],[0.5,1,1,1],[0.9],[1],[0.6],[0.9],[1,0.9,0.9,0.8],[0.2]][model]
BATCH_SIZE = 10000
TEST_USER_BATCH = [4096, 1024][dataset]
SAMPLE_RATE = 1
IF_PRETRAIN = 1
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [2, 5, 10, 20, 50, 100]
GPU_INDEX = "6"

DIR = 'dataset/'+DATASET+'/'
