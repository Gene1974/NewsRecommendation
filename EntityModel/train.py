import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from utils import mrr_score, dcg_score, ndcg_score, EarlyStopping
from . import EntityModel
assert(torch.cuda.is_available())

def encode_all_news(news_encoder, news_info, news_ents = None):
    # print(news_info.shape, news_ents.shape)
    n_news = len(news_info)
    news_rep = []
    n_batch = 32
    for i in range((len(news_info) + n_batch - 1) // n_batch):
        batch_news = torch.tensor(news_info[i * n_batch: (i + 1) * n_batch], dtype = torch.long, device = 'cuda')
        batch_ents = torch.tensor(news_ents[i * n_batch: (i + 1) * n_batch], dtype = torch.long, device = 'cuda')
        # print(batch_news.shape, batch_ents.shape)
        batch_rep = news_encoder(batch_news, batch_ents)
        # print(batch_rep.shape)
        batch_rep = batch_rep.detach().cpu().numpy()
        news_rep.append(batch_rep)
    news_rep = np.concatenate(news_rep, axis = 0)
    return news_rep # (n_news, n_title, n_emb)

def encode_all_user(user_encoder, dev_dataset, user_ids, user_hist, news_rep):
    user_rep = []
    with torch.no_grad():
        for _, batch in enumerate(dev_dataset):
            if len(batch[0]) == 0:
                break
            user_hist_rep = torch.tensor(news_rep[batch[1]], device = 'cuda') # (n_batch, n_hist)
            user = user_encoder(user_hist_rep).detach().cpu().numpy() # (n_batch, emb_dim)
            user_rep.append(user)
    return user_rep # [user_rep, ...]

def train_epoch(model, train_dataset, optimizer, entrophy):
    train_losses = []
    model.train()
    device = 'cuda'
    for _, batch in enumerate(train_dataset):
        if batch[0].shape[0] == 0:
            break
        impr_news, hist_news, hist_labels, impr_ents, hist_ents = batch
        impr_news = torch.tensor(impr_news, dtype = torch.long, device = device)
        hist_news = torch.tensor(hist_news, dtype = torch.long, device = device)
        hist_correct = torch.argmax(torch.tensor(hist_labels, dtype = torch.long, device = device), dim = 1)
        impr_ents = torch.tensor(impr_ents, dtype = torch.long, device = device)
        hist_ents = torch.tensor(hist_ents, dtype = torch.long, device = device)
        optimizer.zero_grad()
        output = model(impr_news, hist_news, impr_ents, hist_ents)
        loss = entrophy(output, hist_correct)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.average(train_losses)

def evaluate(model, dev_dataset, news_info, dev_users, dev_user_hist, news_ents = None):
    news_rep = encode_all_news(model.news_encoder, news_info, news_ents) # (65238, 132) (65238, 5, 11)
    user_rep = encode_all_user(model.user_encoder, dev_dataset, dev_users, dev_user_hist, news_rep)
    
    model.eval()
    with torch.no_grad():
        auc_scores = []
        mrr_scores = []
        ndcg5_scores = []
        ndcg10_scores = []
        for i, batch in enumerate(dev_dataset):
            if len(batch[0]) == 0:
                break
            user = user_rep[i]
            for j in range(len(batch[0])):
                sample = news_rep[batch[0][j]] # (n_imp, emb_dim)
                positive = batch[2][j] # (1, n_imp)

                score = np.matmul(sample, user[j]) # (1, n_imp)
                predict = np.exp(score) / np.sum(np.exp(score))

                auc_scores.append(roc_auc_score(positive, predict))
                mrr_scores.append(mrr_score(positive, predict))
                ndcg5_scores.append(ndcg_score(positive, predict, k = 5))
                ndcg10_scores.append(ndcg_score(positive, predict, k = 10))
    return np.mean(auc_scores), np.mean(mrr_scores), np.mean(ndcg5_scores), np.mean(ndcg10_scores)

def train_and_eval(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents = None, epochs = 4):
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    entrophy = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        begin_time = time.time()
        loss = train_epoch(model, train_dataset, optimizer, entrophy)
        auc, mrr, ndcg5, ndcg10 = evaluate(model, dev_dataset, news_info, dev_users, dev_user_hist, news_ents)
        end_time = time.time()
        print('[epoch {:d}] loss: {:.4f}, AUC: {:.4f}, MRR: {:.4f}, nDCG5:{:.4f}, nDCG10: {:.4f}, Time: {:.2f}'.format(
            epoch + 1, loss, auc, mrr, ndcg5, ndcg10, end_time - begin_time))
    return auc, mrr, ndcg5, ndcg10

def train(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents = None, epochs = 6):
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    entrophy = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        begin_time = time.time()
        loss = train_epoch(model, train_dataset, optimizer, entrophy)
        end_time = time.time()
    auc, mrr, ndcg5, ndcg10 = evaluate(model, dev_dataset, news_info, dev_users, dev_user_hist, news_ents)
    return auc, mrr, ndcg5, ndcg10

# def train_multi_times(args, word_emb, cate_emb, ent_emb, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents = None):
#     print(args)
# #     aucs = []
#     for i in range(5):
#         model = Model(args, word_emb, cate_emb, ent_emb).to('cuda')
#         train_and_eval(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, args['epochs'])
#     aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
#     for i in range(3):
#         model = Model(args, word_emb, cate_emb, ent_emb).to('cuda')
#         auc, mrr, ndcg5, ndcg10 = train_and_eval(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, 6)
#         aucs.append(auc)
#         mrrs.append(mrr)
#         ndcg5s.append(ndcg5)
#         ndcg10s.append(ndcg10)
#     auc, mrr, ndcg5, ndcg10 = np.average(aucs), np.average(mrrs), np.average(ndcg5s), np.average(ndcg10s)
#     print(auc, mrr, ndcg5, ndcg10)
#     print('Average AUC: {:.4f} , MRR: {:.4f}, nDCG5:{:.4f}, nDCG10: {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
#     return auc, mrr, ndcg5, ndcg10




