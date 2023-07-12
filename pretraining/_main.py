## For latent embedding pre-training
## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

from train_model import *
from params import all_para
from print_save import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = all_para['GPU_INDEX']

if __name__ == '__main__':
    ## print model hyperparameters
    print_params(all_para)
    ## train the model
    train_model(all_para)

