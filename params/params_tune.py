## hyper-parameter setting
## author@Wenhui Yu  2023.07.09
## email: jianlin.ywh@alibaba-inc.com

tuning_method = ['tuning', 'fine_tuning', 'cross_tuning', 'coarse_tuning', 'test'][0]  # set here to tune model or test model
lr_coarse, lamda_coarse = 0.001, 0.01       # start coarse search at
lr_fine, lamda_fine = 0.0005, 0.1           # start fine search at, needed in only ``fine_tuning''
min_num_coarse, max_num_coarse = 3, 5       # repeat numbers
min_num_fine, max_num_fine = 10, 20
iter_num_test = 20
