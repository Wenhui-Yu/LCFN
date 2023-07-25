import random
import json

def write_data(path, data):
    f = open(path, 'w')
    jsObj = json.dumps(data)
    f.write(jsObj)
    f.close()

def dataset_filtering(interaction, core):
    # filtering the dataset with core
    # movielens is filtered by only remaining only users with at least 20 interactions
    # we further filter the dataset by remaining users and items with at least 20 interactions
    user_id_dic = {}  # record the number of interaction for each user and item
    item_id_dic = {}
    for (user_id, item_id) in interaction:
        try:
            user_id_dic[user_id] += 1
        except:
            user_id_dic[user_id] = 1
        try:
            item_id_dic[item_id] += 1
        except:
            item_id_dic[item_id] = 1
    print ('#Original dataset')
    print('  User:', len(user_id_dic), 'Item:', len(item_id_dic))
    print('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:',
          100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    sort_user = []
    sort_item = []
    for user_id in user_id_dic:
        sort_user.append((user_id, user_id_dic[user_id]))
    for item_id in item_id_dic:
        sort_item.append((item_id, item_id_dic[item_id]))
    sort_user.sort(key=lambda x: x[1])
    sort_item.sort(key=lambda x: x[1])
    print ('Fitering(core = ', core, '...', end = '')
    while sort_user[0][1] < core or sort_item[0][1] < core:
        # find out all users and items with less than core recorders
        user_LessThanCore = set()
        item_LessThanCore = set()
        for pair in sort_user:
            if pair[1] < core:
                user_LessThanCore.add(pair[0])
            else:
                break
        for pair in sort_item:
            if pair[1] < core:
                item_LessThanCore.add(pair[0])
            else:
                break

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
            try:
                user_id_dic[user_id] += 1
            except:
                user_id_dic[user_id] = 1
            try:
                item_id_dic[item_id] += 1
            except:
                item_id_dic[item_id] = 1

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
    print ('#Filtered dataset')
    print ('  User:', len(user_id_dic), 'Item:', len(item_id_dic), 'Interaction:', len(interaction), 'Sparsity:', 100 - len(interaction) * 100.0 / len(user_id_dic) / len(item_id_dic), '%')
    return interaction

def index_encoding(interaction):
    # mapping in into number
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
    return interaction

def dataset_split(Interaction):
    user_interaction = []
    item_interaction = []
    for interaction in Interaction:
        while len(user_interaction) <= interaction[0]:
            user_interaction.append([])
        while len(item_interaction) <= interaction[1]:
            item_interaction.append(0)
        user_interaction[interaction[0]].append(interaction[1])
        item_interaction[interaction[1]] += 1
    validation_data = []
    test_data = []
    for i in range(len(user_interaction)):
        validation_data.append([])
        test_data.append([])
    #print(item_interaction)
    for i in range(len(user_interaction)):
        interactions = user_interaction[i]
        for ii in range(round(0.1 * len(interactions))):
            item = int(random.uniform(0, len(interactions)))
            while item_interaction[interactions[item]] <= cold_thre:
                item = int(random.uniform(0, len(interactions)))
            item_interaction[interactions[item]] -= 1
            validation_data[i].append(interactions[item])
            interactions.pop(item)
            item = int(random.uniform(0, len(interactions)))
            while item_interaction[interactions[item]] <= cold_thre:
                item = int(random.uniform(0, len(interactions)))
            item_interaction[interactions[item]] -= 1
            test_data[i].append(interactions[item])
            interactions.pop(item)
    #print(item_interaction)
    return user_interaction, validation_data, test_data

core = 20
cold_thre = 15  # to avoid cold/cool item (items with less than `cold_thre' records)
#path_read = 'ratings.dat'
path_read = 'u.data'
path_train = 'train_data.json'
path_test = 'test_data.json'
path_validation = 'validation_data.json'
f = open(path_read, "r")
data = f.read()
f.close()
Interaction = []
data = data.split('\n')
for interactions in data:
    #interactions = interactions.split('::')
    interactions = interactions.split('\t')
    if len(interactions) > 1:
        Interaction.append((interactions[0], interactions[1]))
#Interaction = list(set(Interaction))

#print(Interaction[0:10])
Interaction = dataset_filtering(Interaction, core)
#print(Interaction[0:10])
Interaction = index_encoding(Interaction)
#print(Interaction[0:10])
train_data, validation_data, test_data = dataset_split(Interaction)

write_data(path_train, train_data)
write_data(path_validation, validation_data)
write_data(path_test, test_data)