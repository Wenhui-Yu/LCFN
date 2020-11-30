## split train data into batches and train the model
## author@Wenhui Yu  2020.06.02
## email: yuwh16@mails.tsinghua.edu.cn

from model_BPR import *
from model_NCF import *
from model_GCMC import *
from model_NGCF import *
from model_SCF import *
from model_LCFN import *
from model_CGMC import *
from test_model import *
from read_data import *
from print_save import *
import gc
import time

def train_model(para, path_excel):
    [_,_,MODEL,LR,LAMDA,LAYER,EMB_DIM,FREQUENCY_USER, FREQUENCY_ITEM,
     BATCH_SIZE, SAMPLE_RATE,IF_PRETRAIN,N_EPOCH,_,TOP_K,OPTIMIZATION] = para
    ## Paths of data
    train_path = DIR+'train_data.json'
    transformation_bases_path = DIR+'hypergraph_embeddings.json'                  # transformation bases for graph convolution
    pre_train_feature_path = DIR+'pre_train_feature'+str(EMB_DIM)+'.json'         # to pretrain latent factors for user-item interaction

    ## Load data
    # load training data
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    # load pre-trained embeddings for all deep models
    try:
        pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
    except:
        print('There is no pre-trained feature found!!')
        pre_train_feature = [0, 0]
        IF_PRETRAIN = 0
        
    # load pre-trained transform bases for LCFN
    if MODEL == 'LCFN': transformation_bases = read_bases(transformation_bases_path, FREQUENCY_USER, FREQUENCY_ITEM)

    ## Define the model
    if MODEL == 'BPR':
        model = model_BPR(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                          optimization=OPTIMIZATION)
    if MODEL == 'NCF':
        model = model_NCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                          optimization=OPTIMIZATION, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'GCMC':
        model = model_GCMC(layer=LAYER, graph=train_data_interaction, n_users=user_num, n_items=item_num, 
                           emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, optimization=OPTIMIZATION, 
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'NGCF':
        model = model_NGCF(layer=LAYER, graph=train_data_interaction, n_users=user_num, n_items=item_num, 
                           emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, optimization=OPTIMIZATION,
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'SCF':
        model = model_SCF(layer=LAYER, graph=train_data_interaction, n_users=user_num, n_items=item_num, 
                          emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, optimization=OPTIMIZATION, 
                          pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'CGMC':
        model = model_CGMC(layer=LAYER, graph=train_data_interaction, n_users=user_num, n_items=item_num, 
                           emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, optimization=OPTIMIZATION, 
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'LCFN':
        model = model_LCFN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, 
                           graph_embeddings=transformation_bases, lr=LR, lamda=LAMDA, optimization=OPTIMIZATION, 
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    for epoch in range(N_EPOCH):
        t1 = time.clock()
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                (user, pos_item) = train_data_interaction[sample]
                sample_num = 0
                while sample_num < SAMPLE_RATE:
                    neg_item = int(random.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, pos_item, neg_item])
            train_batch_data = np.array(train_batch_data)
            _, loss = sess.run([model.updates, model.loss],
                               feed_dict={model.users: train_batch_data[:,0],
                                          model.pos_items: train_batch_data[:,1],
                                          model.neg_items: train_batch_data[:,2]})

        # test the model each epoch
        F1, NDCG = test_model(sess, model)
        t2 = time.clock()
        F1_max = max(F1_max, F1[0])
        # print performance
        print_value([epoch + 1, loss, F1_max, F1, NDCG])
        # save performance
        F1_df.loc[epoch + 1] = F1
        NDCG_df.loc[epoch + 1] = NDCG
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if not loss < 10**10:
            break
        
    del model, loss, _, sess
    gc.collect()
