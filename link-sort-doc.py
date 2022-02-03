import numpy as np

f1_name = "ltr_data/search_engine_res.dat"
f2_name = "ltr_data/predictions"

ranking = {}

def rank_array(d):
    res = []
    for _, v in d.values():
        res.append(v)
    return res

def dcg(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

# param r should be the rank found by the model
def calculate_ndcg(r, k):
    ideal_r = dict(sorted(r.items(), key=lambda item: item[1][1], reverse=True))
    return dcg(rank_array(r), k) / dcg(rank_array(ideal_r), k)

with open(f1_name, 'r') as f1, open(f2_name, 'r') as f2:
    index = 0

    for i in range(0, 16):
        l1 = f1.readline()
        l2 = f2.readline()
        ranking[str(index)] = (float(l2), l1[0])
        index += 1

ranking_sorted = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True))
for index, tup in ranking_sorted.items():
    print("Index: {} | Ranking: {} | Label: {}".format(index, tup[0], tup[1]))

print("\nNDCG@{} encontrado: {}".format(5, calculate_ndcg(ranking_sorted, 5)))
print("NDCG@{} encontrado: {}".format(10, calculate_ndcg(ranking_sorted, 10)))
print("NDCG@{} encontrado: {}".format(15, calculate_ndcg(ranking_sorted, 15)))
