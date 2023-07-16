## author @Wenhui Yu  2021.01.24
## split train data into batches and train the model

from models.model_MF import model_MF
from models.model_NCF import model_NCF
from models.model_NGCF import model_NGCF
from models.model_LightGCN import model_LightGCN
from models.model_LGCN import model_LGCN
from utils.test_model import test_model
from utils.print_save import print_value, save_value
import tensorflow as tf
import numpy as np
import random as rd
import pandas as pd

def train_model(para, data, path_excel):
    ## data and hyperparameters
    [train_data, train_data_interaction, popularity, user_num, item_num, test_data, pre_train_embeddings,
     graph_embeddings, sparse_propagation_matrix] = data
    para_test = [train_data, test_data, user_num, item_num, para['TOP_K'], para['TEST_USER_BATCH']]
    data = {'user_num': user_num, "item_num": item_num, "popularity": popularity, "pre_train_embeddings": pre_train_embeddings,
            "graph_embeddings": graph_embeddings, "sparse_propagation_matrix": sparse_propagation_matrix}
    ## Define the model
    if para["MODEL"] == 'MF': model = model_MF(data=data, para=para)
    if para["MODEL"] == 'NCF': model = model_NCF(data=data, para=para)
    if para["MODEL"] == 'NGCF': model = model_NGCF(data=data, para=para)
    if para["MODEL"] == 'LightGCN': model = model_LightGCN(data=data, para=para)
    if para["MODEL"] == 'LGCN': model = model_LGCN(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), para["BATCH_SIZE"]))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=para["TOP_K"])
    NDCG_df = pd.DataFrame(columns=para["TOP_K"])
    for epoch in range(para["N_EPOCH"]):
        for batch_num in range(len(batches) - 1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num + 1]):
                (user, pos_item) = train_data_interaction[sample]
                sample_num = 0
                while sample_num < para["SAMPLE_RATE"]:
                    neg_item = int(rd.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, pos_item, neg_item])
            train_batch_data = np.array(train_batch_data)
            _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2]})
        ## test the model each epoch
        F1, NDCG = test_model(sess, model, para_test)
        F1_max = max(F1_max, F1[1])     # tuning model according to f1@20
        ## print performance
        # print_value([epoch + 1, loss, F1_max, F1, NDCG])
        if epoch % 10 == 0: print('%.5f' % (F1_max), end = ' ', flush = True)
        ## save performance
        F1_df.loc[epoch + 1] = F1
        NDCG_df.loc[epoch + 1] = NDCG
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if loss > 10 ** 10: break
    print()
    return F1_max