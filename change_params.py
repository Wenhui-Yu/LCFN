def change_params(all_para, change_dic, pred_dim):
    para_name2para_id = {'GPU_INDEX': 0, 'DATASET': 1, 'MODEL': 2, 'LR': 3, 'LAMDA': 4, 'LAYER': 5, 'EMB_DIM': 6,
                         'BATCH_SIZE': 7, 'TEST_USER_BATCH': 8, 'N_EPOCH': 9, 'IF_PRETRAIN': 10, 'TEST_VALIDATION': 11,
                         'TOP_K': 12, 'FREQUENCY_USER': 13, 'FREQUENCY_ITEM': 14, 'FREQUENCY': 15, 'KEEP_PORB': 16,
                         'SAMPLE_RATE': 17, 'GRAPH_CONV': 18, 'PREDICTION': 19, 'LOSS_FUNCTION': 20,
                         'GENERALIZATION': 21, 'OPTIMIZATION': 22, 'IF_TRASFORMATION': 23, 'ACTIVATION': 24,
                         'POOLING': 25, 'PROP_DIM': 26, 'PROP_EMB': 27, 'IF_NORM': 28}
    for para in change_dic:
        if para not in ['model', 'dataset', 'test_validation', 'pred_dim', 'dic_end']:
            all_para[para_name2para_id[para]] = change_dic[para]
    dataset = {'Amazon': 0, 'Movielens': 1}[all_para[1]]
    model = {'MF': 0, 'NCF': 1, 'GCMC': 2, 'NGCF': 3, 'SCF': 4, 'CGMC': 5, 'LightGCN': 6, 'LCFN': 7, 'LightLCFN': 8, 'SGNN': 9}[all_para[2]]
    for para in change_dic:
        if para == 'dataset':
            dataset = change_dic[para]
            all_para[1] = ['Amazon', 'Movielens', 'Movielens_large'][change_dic[para]]
        if para == 'model':
            model = change_dic[para]
            all_para[2] = ['MF', 'NCF', 'GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN', 'LCFN', 'LightLCFN', 'SGNN'][change_dic[para]]
        if para == 'test_validation': all_para[11] = ['Validation', 'Test'][change_dic[para]]
        if para == 'pred_dim': pred_dim = change_dic[para]
    all_para[3] = [[0.05, 0.0002, 0.001, 0.0001, 0.0001, 0.0001, 0.005, 0.0005, 0.0005, 0.0005],
                   [0.02, 0.00001, 0.0002, 0.00005, 0.0001, 0.00002, 0.0005, 0.0005, 0.0005, 0.0005],
                   [0.05, 0.0002, 0.001, 0.0001, 0.0001, 0.0001, 0.005, 0.0005, 0.0005, 0.0005]][dataset][model]
    all_para[4] = [[0.02, 0, 0.05, 0.001, 0.02, 0.0002, 0.02, 0.005, 0.02, 0.02],
                   [0.01, 0, 0.02, 0.02, 0.01, 0.05, 0.02, 0.01, 0.05, 0.05],
                   [0.02, 0, 0.05, 0.001, 0.02, 0.0002, 0.02, 0.005, 0.02, 0.02]][dataset][model]
    all_para[5] = [[0, 4, 1, 1, 1, 1, 2, 1, 2, 2], [0, 4, 1, 1, 1, 1, 2, 1, 2, 2], [0, 4, 1, 1, 1, 1, 2, 1, 2, 2]][dataset][model]
    all_para[6] = [pred_dim, int(pred_dim / 2), int(pred_dim / (all_para[5] + 1)), int(pred_dim / (all_para[5] + 1)), int(pred_dim / (all_para[5] + 1)),
                   int(pred_dim / (all_para[5] + 1)), pred_dim, int(pred_dim / (all_para[5] + 1)), pred_dim, pred_dim][model]
    all_para[8] = [4096, 1024, 16384][dataset]
    all_para[13] = [100, 300, 200][dataset]
    all_para[14] = [50, 200, 40][dataset]
    ## hyperparameters for LightLCFN
    if all_para[25] == 'Concat': all_para[6] = int(pred_dim / (all_para[5] + 1))
    for para in change_dic:
        if para not in ['model', 'dataset', 'test_validation', 'pred_dim', 'dic_end']:
            all_para[para_name2para_id[para]] = change_dic[para]
    return all_para