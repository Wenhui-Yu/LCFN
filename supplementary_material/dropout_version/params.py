## hyper-parameter setting
## author@Wenhui Yu  2020.06.02
## email: yuwh16@mails.tsinghua.edu.cn

model = 6           # 0:BPR, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:LCFN (select the model, 0-5 baselines)
dataset = 0         # 0:Amazon, 1:Movielens (select the dataset)
validate_test = 1   # 0:Validate, 1: Test (Validation set for model tuning and test set for testing)
DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = ['BPR', 'NCF', 'GCMC', 'NGCF', 'SCF', 'CGMC', 'LCFN'][model]
OPTIMIZATION = ['SGD', 'SGD', 'Adam', 'Adam', 'RMSProp', 'Adam', 'Adam'][model]
LR = [[0.05,0.0002,0.001,0.0001,0.0001,0.0001,0.0005], [0.02,0.00001,0.0002,0.00005,0.0001,0.00002,0.0005]][dataset][model]
LAMDA = [[0.02,0,0.05,0.001,0.02,0.0002,0.005], [0.01,0,0.02,0.02,0.01,0.05,0.01]][dataset][model]
LAYER = [[0,4,1,1,1,1,1], [0,4,1,1,1,1,1]][dataset][model]
pred_dim = 128 # predictive embedding dimensionality
# embedding layer dimensionality
EMB_DIM = [pred_dim,int(pred_dim/2),int(pred_dim/(LAYER+1)),int(pred_dim/(LAYER+1)),
           int(pred_dim/(LAYER+1)),int(pred_dim/(LAYER+1)),int(pred_dim/(LAYER+1))][model]
FREQUENCY_USER = [[0,0,0,0,0,0,100], [0,0,0,0,0,0,300]][dataset][model]
FREQUENCY_ITEM = [[0,0,0,0,0,0,50], [0,0,0,0,0,0,200]][dataset][model]
KEEP_RATE = [[0.8],[0.5,1,1,1],[0.9],[1],[0.6],[0.9],[1,0.9,0.9,0.8]][model]
BATCH_SIZE = 10000
TEST_USER_BATCH = [4096, 1024][dataset]
SAMPLE_RATE = 1
IF_PRETRAIN = 1
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][validate_test]
TOP_K = [2, 5, 10, 20, 50, 100]
GPU_INDEX = "0"

DIR = 'dataset/'+DATASET+'/'
