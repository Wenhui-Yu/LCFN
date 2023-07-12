## split train data into batches and train the model
## author@Wenhui Yu  2020.06.02
## email: jianlin.ywh@alibaba-inc.com

from model_MF import *
from test_model import *
from print_save import *
from params import DIR

def train_model(para):
    ## paths of data
    train_path = DIR + 'train_data.json'
    validation_path = DIR + 'validation_data.json'
    save_embeddings_path = DIR + 'pre_train_embeddings' + str(para['EMB_DIM']) + '.json'

    ## Load data
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    test_data = read_data(validation_path)[0]
    para_test = [train_data, test_data, user_num, item_num, para['TOP_K'], para['TEST_USER_BATCH']]

    data = {'user_num': user_num, "item_num": item_num, "popularity": 0, "pre_train_embeddings": 0,
            "graph_embeddings": 0, "sparse_propagation_matrix": 0}

    ## define the model
    model = model_MF(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data_interaction), para['BATCH_SIZE']))
    batches.append(len(train_data_interaction))

    ## training iteratively
    F1_max = 0
    for epoch in range(para['N_EPOCH']):
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                (user, pos_item) = train_data_interaction[sample]
                sample_num = 0
                while sample_num < para['SAMPLE_RATE']:
                    neg_item = int(random.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, pos_item, neg_item])
            train_batch_data = np.array(train_batch_data)
            _, loss = sess.run([model.updates, model.loss],
                               feed_dict={model.users: train_batch_data[:,0],
                                          model.pos_items: train_batch_data[:,1],
                                          model.neg_items: train_batch_data[:,2]})
        F1, NDCG = test_model(sess, model, para_test)
        if F1[1] > F1_max:
            F1_max = F1[1]
            user_embeddings, item_embeddings = sess.run([model.user_embeddings, model.item_embeddings])
        ## print performance
        print_value([epoch + 1, loss, F1_max, F1, NDCG])
        if not loss < 10 ** 10:
            break
    save_embeddings([user_embeddings.tolist(), item_embeddings.tolist()], save_embeddings_path)