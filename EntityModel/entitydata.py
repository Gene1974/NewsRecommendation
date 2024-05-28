import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
assert(torch.cuda.is_available())

# load pre-trained entity embedding
def load_ent_emb(ent_dict):
    ent_emb = np.random.rand(len(ent_dict), 100)
    with open('/data/Recommend/MIND/large_entity_embedding.vec', 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            ent_id = data[0]
            if ent_id in ent_dict:
                ent_emb[ent_dict[ent_id]] = [float(i) for i in data[1:]]
    ent_emb = torch.tensor(ent_emb, dtype = torch.float)
    print(ent_emb.shape)
    return ent_emb

# load triplets from kg
def load_triplets(ent_dict):
    rel_dict = {}
    triplets = {ent: [] for ent in ent_dict}
    with open('/data/Recommend/MIND/wikidata_graph.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h in triplets:
                if r not in rel_dict:
                    rel_dict[r] = len(rel_dict)
                if t not in ent_dict:
                    ent_dict[t] = len(ent_dict)
                triplets[h].append([ent_dict[h], rel_dict[r], ent_dict[t]])
    print(triplets)
    return ent_dict, rel_dict, triplets


def map_triplets(triplets, ent_emb):
    trip_tails = [] # (n_ent, n_neighbor)
    trip_embs = []
    for key in triplets:
        trip = triplets[key]
        tails = [t[2] for t in trip]
        trip_tails.append(tails)
    return trip_tails


# load triplets
def load_triplets(ent_dict):
    rel_dict = {}
    triplets = {ent: [] for ent in ent_dict}
    print('original entities:', len(ent_dict))
    with open('/data/Recommend/MIND/wikidata_graph.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h in triplets:
                if r not in rel_dict:
                    rel_dict[r] = len(rel_dict)
                if t not in ent_dict:
                    ent_dict[t] = len(ent_dict)
                triplets[h].append([ent_dict[h], rel_dict[r], ent_dict[t]])
    print('load graph: ', len(rel_dict), len(triplets))
    return ent_dict, rel_dict, triplets # 127447 760 32474

def check_ent_num():
    from data import load_news
    news_list, newsid_dict, word_dict, cate_dict, ent_dict = load_news('/data/Recommend/MIND/small_news.tsv')
    print(len(ent_dict)) # 65238
    ent_dict, rel_dict, triplets = load_triplets(ent_dict)
    print(len(ent_dict))

if __name__ == '__main__':
    check_ent_num()
    

