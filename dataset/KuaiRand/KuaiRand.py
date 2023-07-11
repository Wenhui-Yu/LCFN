import pandas as pd

import random
import json


def write_data(path, data):
    f = open(path, 'w')
    jsObj = json.dumps(data)
    f.write(jsObj)
    f.close()

def dataset_filtering(interaction, core):
    # filter the cold users and items within 10 interactions
    user_id_dic = {}  # record the number of interaction for each user and item
    item_id_dic = {}
    for (user_id, item_id) in interaction:
        try: user_id_dic[user_id] += 1
        except: user_id_dic[user_id] = 1
        try: item_id_dic[item_id] += 1
        except: item_id_dic[item_id] = 1
    print ('# Original training dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    sort_user = []
    sort_item = []
    for user_id in user_id_dic:
        sort_user.append((user_id, user_id_dic[user_id]))
    for item_id in item_id_dic:
        sort_item.append((item_id, item_id_dic[item_id]))
    sort_user.sort(key=lambda x: x[1])
    sort_item.sort(key=lambda x: x[1])
    print ('Fitering (core = ' + str(core) + ') ... ', end = 'number of remained interactions: ')
    while sort_user[0][1] < core or sort_item[0][1] < core:
        # find out all users and items with less than core recorders
        user_LessThanCore = set()
        item_LessThanCore = set()
        for pair in sort_user:
            if pair[1] < core: user_LessThanCore.add(pair[0])
            else: break
        for pair in sort_item:
            if pair[1] < core: item_LessThanCore.add(pair[0])
            else: break
        # reconstruct the interaction record, remove the cool one
        interaction_filtered = []
        for (user_id, item_id) in interaction:
            if not (user_id in user_LessThanCore or item_id in item_LessThanCore):
                interaction_filtered.append((user_id, item_id))
        # update the record
        interaction = interaction_filtered
        # count the number of each user and item in new data, check if all cool users and items are removed
        # reset all memory variables
        user_id_dic = {}  # record the number of interaction for each user and item
        item_id_dic = {}
        for (user_id, item_id) in interaction:
            try: user_id_dic[user_id] += 1
            except: user_id_dic[user_id] = 1
            try: item_id_dic[item_id] += 1
            except: item_id_dic[item_id] = 1

        sort_user = []
        sort_item = []
        for user_id in user_id_dic:
            sort_user.append((user_id, user_id_dic[user_id]))
        for item_id in item_id_dic:
            sort_item.append((item_id, item_id_dic[item_id]))
        sort_user.sort(key=lambda x: x[1])
        sort_item.sort(key=lambda x: x[1])
        print (len(interaction), end = ' ')
    print()
    print ('# Filtered training dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    return interaction

def index_encoding(interaction):
    # mapping id into number
    # after filtering the dataset, we need to re-encode the index of users and items
    user_id_set = set()
    item_id_set = set()

    for (user_id, item_id) in interaction:
        user_id_set.add(user_id)
        item_id_set.add(item_id)
    user_num2id = list(user_id_set)
    item_num2id = list(item_id_set)
    user_num2id.sort()
    item_num2id.sort()
    # user_id2num maps id to number, and user_num2id dictionary is not needed, user_ID
    user_id2num = {}
    for num in range(0, len(user_id_set)):
        user_id2num[user_num2id[num]] = num
    item_id2num = {}
    for num in range(0, len(item_id_set)):
        item_id2num[item_num2id[num]] = num
    interaction_number = []
    for (user_id, item_id) in interaction:
        interaction_number.append((user_id2num[user_id], item_id2num[item_id]))
    interaction = interaction_number
    return interaction, user_id2num, item_id2num

def get_popularity(Interaction):
    item_num = 0
    for interaction in Interaction:
        item_num = max(item_num, interaction[1])
    popularity = []
    for i in range(item_num + 1):
        popularity.append(0)
    for interaction in Interaction:
        popularity[interaction[1]] += 1
    return popularity

def dataset_split(Interaction_train, Interaction_vali_test, user_id2num, item_id2num):
    user_num = len(user_id2num)
    train_data = []
    validation_data = []
    test_data = []
    for i in range(user_num):
        train_data.append([])
        validation_data.append([])
        test_data.append([])
    for interaction in Interaction_train:
        train_data[interaction[0]].append(interaction[1])
    for interaction in Interaction_vali_test:
        if interaction[0] in user_id2num and interaction[1] in item_id2num:
            if int(interaction[2]) < 20220500:
                validation_data[user_id2num[interaction[0]]].append(item_id2num[interaction[1]])
            else:
                test_data[user_id2num[interaction[0]]].append(item_id2num[interaction[1]])
    return train_data, validation_data, test_data

def count_data(Interation_train, Interaction_test):
    user_set = []
    item_set = []
    for interaction in Interation_train:
        user_set.append(interaction[0])
        item_set.append(interaction[1])
    for interaction in Interaction_test:
        user_set.append(interaction[0])
        item_set.append(interaction[1])
    user_set = list(set(user_set))
    item_set = list(set(item_set))
    print("# All data (train + validate + test)")
    print("  User: ", len(user_set), "Item: ", len(item_set), "Interaction: ", len(Interation_train) + len(Interaction_test))

core = 10
path_train_read = 'KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv'
path_test_read = 'KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv'
path_train = 'train_data.json'
path_test = 'test_data.json'
path_validation = 'validation_data.json'
path_popularity = 'popularity.json'

print("reading data...")
data_train = []
raw_data_train = pd.read_csv(path_train_read, usecols=['user_id', 'video_id', 'is_click'])
row_num = len(raw_data_train['user_id'])
for i in range(row_num):
    if raw_data_train['is_click'][i] == 1:
        data_train.append((raw_data_train['user_id'][i], raw_data_train['video_id'][i]))
data_test = []
raw_data_test = pd.read_csv(path_test_read, usecols=['user_id', 'video_id', 'is_click', 'date'])
row_num = len(raw_data_test['user_id'])
for i in range(row_num):
    if raw_data_test['is_click'][i] == 1:
        data_test.append((raw_data_test['user_id'][i], raw_data_test['video_id'][i], raw_data_test['date'][i]))

print("processing data...")
count_data(data_train, data_test)
data_train = list(set(data_train))
data_train = dataset_filtering(data_train, core)
data_train, user_id2num, item_id2num = index_encoding(data_train)
popularity = get_popularity(data_train)
data_test = list(set(data_test))
data_train, validation_data, test_data = dataset_split(data_train, data_test, user_id2num, item_id2num)

print("saving data...")
write_data(path_train, data_train)
write_data(path_validation, validation_data)
write_data(path_test, test_data)
write_data(path_popularity, popularity)
