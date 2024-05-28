import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
assert(torch.cuda.is_available())

def clear_str(text):
    return text.lower().replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('\'', '').replace('"', '').replace('?', '').replace('!', '').replace('(', '').replace(')', '').split(' ')

def load_news(path):
    news_dict = {} # index -> news
    news_list = [] # index -> news
    newsid_dict = {} # newsid -> index
    word_dict = {'<PAD>': 0, '<OOV>': 1}
    cate_dict = {'<PAD>': 0, '<OOV>': 1}
    ent_dict = {'<PAD>': 0, '<OOV>': 1}
    
    with open(path, 'r') as f:
        for line in f.readlines():
            news_id, category, subcategory, title, abstract, \
                url, title_entities, abstract_entities = line.strip().split('\t')
            title = clear_str(title)
            abstract = clear_str(abstract)
            # todo: title_entities only
            entities = json.loads(title_entities) + json.loads(abstract_entities)
            ent_words = [clear_str(ent['Label']) for ent in entities] # list(list(str))
            ent_ids = [ent['WikidataId'] for ent in entities] # list(str): ['Q80976', 'Q43274', 'Q9682']
            for word in title + abstract:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)
            for item in ent_words:
                for word in item:
                    if word not in word_dict:
                        word_dict[word] = len(word_dict)
            for ent in ent_ids:
                if ent not in ent_dict:
                    ent_dict[ent] = len(ent_dict)
            if category not in cate_dict:
                cate_dict[category] = len(cate_dict)
            if subcategory not in cate_dict:
                cate_dict[subcategory] = len(cate_dict)
            if news_id not in newsid_dict:
                newsid_dict[news_id] = len(newsid_dict)
                news_list.append([category, subcategory, title, abstract, ent_words, ent_ids])
    print('load_news: news:{} word:{} cate:{} ent:{}'.format(len(news_list), len(word_dict), len(cate_dict), len(ent_dict)))
    return news_list, newsid_dict, word_dict, cate_dict, ent_dict

# news_info: news_index -> title
# news_ents: news_index -> news_ents(id, words, trips)
def map_news_input(news_list, word_dict, cate_dict, ent_dict, trip_data = None):
    max_title = 30
    max_body = 100
    max_ent_words = 10 # 每个entity最多保留10个单词
    max_ent_trips = 9 # 每个entity最多9个单跳邻居
    max_ent = 5 # 每条新闻最多保留5个entity
    n_news = len(news_list)
    titles = np.zeros((n_news, max_title), dtype = 'int32')
    bodys = np.zeros((n_news, max_body), dtype = 'int32')
    cates = np.zeros((n_news,1), dtype = 'int32')
    subcates = np.zeros((n_news,1), dtype = 'int32')
    ent_words = np.zeros((n_news, max_ent, max_ent_words), dtype = 'int32')
    ent_ids = np.zeros((n_news, max_ent), dtype = 'int32')
    ent_trips = np.zeros((n_news, max_ent, max_ent_trips), dtype = 'int32') # news_index -> news_ent_trips
    for i in range(n_news):
        category, subcategory, title, abstract, ent_word, ent_id = news_list[i]
        titles[i, :len(title)] = [word_dict[word] for word in title[:max_title]]
        bodys[i, :len(abstract)] = [word_dict[word] for word in abstract[:max_body]]
        cates[i] = cate_dict[category]
        subcates[i] = cate_dict[subcategory]
        ent_ids[i, :len(ent_id)] = [ent_dict[ent] for ent in ent_id[:max_ent]]
        for j in range(min(len(ent_word), max_ent)):
            ent_words[i, j, :len(ent_word[j])] = [word_dict[word] for word in ent_word[j][:max_ent_words]]
        # TODO: 直接截断会不会有问题
        ent_trip = [trip_data[ent_id] for ent_id in ent_ids[i]]
        ent_trip_tail = [[trip[2] for trip in trip_data[ent_id]] for ent_id in ent_ids[i]]
        for j in range(min(len(ent_trip_tail), max_ent)):
            ent_trips[i, j, :len(ent_trip_tail[j])] = [ent for ent in ent_trip_tail[j][:max_ent_trips]]
    news_info = np.concatenate((titles, bodys, cates, subcates), axis = 1)
    news_ents = np.concatenate((ent_ids.reshape(n_news, max_ent, 1), ent_words, ent_trips), axis = 2)
    print('map_news_input: news_info: {} news_ents: {}'.format(news_info.shape, news_ents.shape))
    return news_info, news_ents # index -> news_info

def load_glove(word_to_ix, dim = 100):
    if dim == 100:
        path = '/data/pretrained/Glove/glove.6B.100d.txt'
    elif dim == 300:
        path = '/data/pretrained/Glove/glove.840B.300d.txt'
    word_emb = []
    word_emb = np.zeros((len(word_to_ix), dim), dtype = float)
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(' ') # [word emb1 emb2 ... emb n]
            word = data[0]
            if word in word_to_ix:
                word_emb[word_to_ix[word]] = [float(i) for i in data[1:]]
    print('load glove: ',word_emb.shape)
    return torch.tensor(word_emb, dtype = torch.float)

def load_train_impression(path, newsid_dict): # train&dev
    logs = []
    with open(path, 'r') as f:
        for line in f:
            imp_id, user_id, time, history, impression = line.strip().split('\t')
            if history:
                history = [newsid_dict[news_id] for news_id in history.split(' ')]
            else:
                history = []
            positive = []
            negative = []
            for item in impression.split(' '):
                news_id, num = item.split('-')
                if num == '1':
                    positive.append(newsid_dict[news_id])
                else:
                    negative.append(newsid_dict[news_id])
            logs.append([history, positive, negative]) # indexs
    return logs

# user_hist: user_index -> user_hist
def map_user(logs): # index -> history, 用 index 代表 user_id, train&dev
    max_history = 50
    n_user = len(logs)
    user_hist = np.zeros((n_user, max_history), dtype = 'int32') # index -> history
    for i in range(n_user):
        history, positive, negative = logs[i]
        n_hist = len(history)
        if n_hist == 0:
            continue
        user_hist[i, -n_hist:] = history[-max_history:]
    return user_hist   

def neg_sample(negative):
    neg_ratio = 4
    if len(negative) < neg_ratio:
        return random.sample(negative * (neg_ratio // len(negative) + 1), neg_ratio)
    else:
        return random.sample(negative, neg_ratio)

# imps: user_index -> user_imp_ids
def get_train_input(logs): # 和 map_user 使用同一个 log
    neg_ratio = 4
    all_pos = [] # 每个 sample 的 pos
    all_neg = []
    user_id = [] # 每个 sample 的 user，用 index 表示，和 map_user 的结果对应
    for i in range(len(logs)):
        history, positive, negative = logs[i]
        for pos in positive:
            all_pos.append(pos)
            all_neg.append(neg_sample(negative))
            user_id.append(i)
    n_imps = len(all_pos)
    imps = np.zeros((n_imps, 1 + neg_ratio), dtype = 'int32')
    for i in range(len(all_pos)):
        imps[i, 0] = all_pos[i]
        imps[i, 1:] = all_neg[i]
    user_id = np.array(user_id, dtype = 'int32')
    labels = np.zeros((n_imps, 1 + neg_ratio), dtype = 'int32')
    labels[:, 0] = 1
    print('get_train_input:', n_imps)
    return imps, user_id, labels

def get_dev_input(logs): # 和 map_user 使用同一个 log
    imps = []
    labels = []
    user_id = np.zeros((len(logs)), dtype = 'int32') # 每个 sample 的 user index，和 map_user 的结果对应
    for i in range(len(logs)):
        history, positive, negative = logs[i]
        imps.append(np.array(positive + negative, dtype = 'int32'))
        labels.append([1] * len(positive) + [0] * len(negative))
        user_id[i] = i
    print('get_dev_input', len(logs))
    return imps, user_id, labels

class TrainDataset(Dataset):
    def __init__(self, imp_datas, imp_users, imp_labels, news_info, user_clicks, batch_size, news_ents = None, news_urls = None):
        self.n_data = imp_datas.shape[0]
        self.imp_datas = imp_datas # (n_imps, 1 + k)
        self.imp_users = imp_users
        self.imp_labels = imp_labels
        self.news = news_info
        self.user_clicks = user_clicks
        self.batch_size = batch_size
        self.news_ents = news_ents
        self.news_urls = news_urls
        
    def __len__(self):
        return int(np.ceil(self.n_data / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_data)
        
        impr_id = self.imp_datas[start: end] # (n_batch, 1 + k)
        impr_news = self.news[impr_id] # (n_batch, 1 + k, news_len)
        user_id = self.imp_users[start: end] # (n_batch)
        hist_news_id = self.user_clicks[user_id] # (n_batch, n_hist)
        hist_news = self.news[hist_news_id] # (n_batch, n_hist, news_len)
        hist_labels = self.imp_labels[start: end] # (n_batch, 1 + k)
        
        if self.news_ents is not None:
            impr_ents = self.news_ents[impr_id]
            hist_ents = self.news_ents[hist_news_id]
            return impr_news, hist_news, hist_labels, impr_ents, hist_ents
        
        if self.news_urls is not None:
            impr_urls = self.news_urls[impr_id]
            hist_urls = self.news_urls[hist_news_id]
            return impr_news, hist_news, hist_labels, impr_urls, hist_urls
        
        return impr_news, hist_news, hist_labels
    
class DevDataset(Dataset): # data 和 label 是 list，每条数据不同长度
    def __init__(self, imp_datas, imp_users, imp_labels, news_info, user_clicks, batch_size):
        self.imp_datas = imp_datas # [imp1, imp2, ..., impn]
        self.imp_users = imp_users # (n_imps)
        self.imp_labels = imp_labels
        self.news = news_info
        self.user_clicks = user_clicks
        self.batch_size = batch_size
        
        self.n_data = len(imp_datas)
        
    def __len__(self):
        return int(np.ceil(self.n_data / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_data)
        
        data_ids = []
        data_news = [] # [(n_imp, news_len)]
        labels = [] # [(n_imp)]
        for i in range(start, end):
            data_id = self.imp_datas[i] # (n_imp)
            data_ids.append(data_id)
            # data_news.append(self.news[data_id]) # (n_imp, news_len)
            labels.append(self.imp_labels[i]) # (n_imp)
        user_id = self.imp_users[start: end] # (n_batch)
        user_news_id = self.user_clicks[user_id] # (n_batch, n_hist)
        # user_news = self.news[user_news_id] # (n_batch, n_hist, news_len)
        
        #return data_news, user_news, labels
        return data_ids, user_news_id, labels

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
    print('load_ent_emb: ', ent_emb.shape)
    return ent_emb

# load triplets from kg
def load_triplets(ent_dict):
    rel_dict = {}
    triplets = {ent_dict[ent]: [] for ent in ent_dict}
    with open('/data/Recommend/MIND/wikidata_graph.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h in ent_dict and ent_dict[h] in triplets:
                if r not in rel_dict:
                    rel_dict[r] = len(rel_dict)
                if t not in ent_dict:
                    ent_dict[t] = len(ent_dict)
                triplets[ent_dict[h]].append([ent_dict[h], rel_dict[r], ent_dict[t]])
    print('load_triplet: ent: {} rel: {} trip: {}'.format(len(ent_dict), len(rel_dict), len(triplets)))
    return ent_dict, rel_dict, triplets


def map_triplets(triplets, ent_emb):
    trip_tails = [] # (n_ent, n_neighbor)
    trip_embs = []
    for key in triplets:
        trip = triplets[key]
        tails = [t[2] for t in trip]
        trip_tails.append(tails)
    return trip_tails

'''
news_list: original news
news_info: mapped news(word ids)
'''
def get_data():
    # get news
    news_list, newsid_dict, word_dict, cate_dict, ent_dict = load_news('/data/Recommend/MIND/small_news.tsv') # ent: 65238
    # get kg triplets
    ent_dict, rel_dict, triplets = load_triplets(ent_dict) # 127447 760 32474
    # encode news
    news_info, news_ents = map_news_input(news_list, word_dict, cate_dict, ent_dict, triplets)
    # pretrained embeddings
    ent_emb = load_ent_emb(ent_dict)
    word_emb = load_glove(word_dict, 300)
    cate_emb = load_glove(cate_dict, 100)
    
    # get user impression history
    train_logs = load_train_impression('/data/Recommend/MIND/MINDsmall_train/behaviors.tsv', newsid_dict)
    dev_logs = load_train_impression('/data/Recommend/MIND/MINDsmall_dev/behaviors.tsv', newsid_dict)
    # get user history
    train_user_hist = map_user(train_logs)
    dev_user_hist = map_user(dev_logs)
    # encode user history
    train_datas, train_users, train_labels = get_train_input(train_logs)
    dev_datas, dev_users, dev_labels = get_dev_input(dev_logs)
    # # valid_datas, valid_users, valid_labels = get_train_input(dev_logs) # 用 train 的方法构造 dev_set

    # get dataset
    train_dataset = TrainDataset(train_datas, train_users, train_labels, news_info, train_user_hist, 16, news_ents)
    dev_dataset = DevDataset(dev_datas, dev_users, dev_labels, news_info, dev_user_hist, 64)
    # # valid_dataset = TrainDataset(valid_datas, valid_users, valid_labels, news_info, dev_user_hist, 16, news_ents)

    return train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, \
        word_emb, cate_emb, ent_emb

if __name__ == '__main__':
    get_data()
