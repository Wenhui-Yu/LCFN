from params import MODEL_list
from params import LR_list
from params import LAMDA_list
from params import LAYER_list
from params import EMB_list
from params import TEST_USER_BATCH_list
from params import FREQUENCY_USER_list
from params import FREQUENCY_ITEM_list

def change_params(all_para, change_dic, pred_dim):
    para_name2para_id = {'GPU_INDEX': 0, 'DATASET': 1, 'MODEL': 2, 'LR': 3, 'LAMDA': 4, 'LAYER': 5, 'EMB_DIM': 6,
                         'BATCH_SIZE': 7, 'TEST_USER_BATCH': 8, 'N_EPOCH': 9, 'IF_PRETRAIN': 10, 'TEST_VALIDATION': 11,
                         'TOP_K': 12, 'FREQUENCY_USER': 13, 'FREQUENCY_ITEM': 14, 'FREQUENCY': 15, 'KEEP_PORB': 16,
                         'SAMPLE_RATE': 17, 'GRAPH_CONV': 18, 'PREDICTION': 19, 'LOSS_FUNCTION': 20,
                         'GENERALIZATION': 21, 'OPTIMIZATION': 22, 'IF_TRASFORMATION': 23, 'ACTIVATION': 24,
                         'POOLING': 25, 'PROP_DIM': 26, 'PROP_EMB': 27, 'IF_NORM': 28}
    for para in change_dic:
        if para not in ['model', 'dataset', 'test_validation', 'pred_dim']:
            all_para[para_name2para_id[para]] = change_dic[para]
    dataset = {'Amazon': 0, 'Movielens': 1}[all_para[1]]
    model = {MODEL_list[i]: i for i in range(len(MODEL_list))}[all_para[2]]
    for para in change_dic:
        if para == 'dataset':
            dataset = change_dic[para]
            all_para[1] = ['Amazon', 'Movielens'][change_dic[para]]
        if para == 'model':
            model = change_dic[para]
            all_para[2] = MODEL_list[change_dic[para]]
        if para == 'test_validation': all_para[11] = ['Validation', 'Test'][change_dic[para]]
        if para == 'pred_dim': pred_dim = change_dic[para]
    all_para[3] = LR_list[dataset][model]
    all_para[4] = LAMDA_list[dataset][model]
    all_para[5] = LAYER_list[dataset][model]
    all_para[6] = EMB_list[model]
    all_para[8] = TEST_USER_BATCH_list[dataset]
    all_para[13] = FREQUENCY_USER_list[dataset]
    all_para[14] = FREQUENCY_ITEM_list[dataset]
    ## hyperparameters for LGCN
    if all_para[25] == 'Concat': all_para[6] = int(pred_dim / (all_para[5] + 1))
    for para in change_dic:
        if para not in ['model', 'dataset', 'test_validation', 'pred_dim']:
            all_para[para_name2para_id[para]] = change_dic[para]
    return all_para