from numpy import *

def evaluation_F1(order, top_k, positive_item):
    epsilon = 0.1 ** 10
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / max(len(top_k_items), epsilon)
    recall = len(top_k_items & positive_item) / max(len(positive_item), epsilon)
    F1 = 2 * precision * recall / max(precision + recall, epsilon)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    epsilon = 0.1**10
    DCG = 0
    iDCG = 0 
    for i in range(top_k):
        if top_k_item[i] in positive_item:
            DCG += 1 / log2(i + 2)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / log2(i + 2)
    NDCG = DCG / max(iDCG, epsilon)
    return NDCG

