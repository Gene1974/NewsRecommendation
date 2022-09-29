import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchtools import EarlyStopping
from AttentionModel import AttentionPooling, ScaledDotProductAttention, MultiHeadSelfAttention
assert(torch.cuda.is_available())

# Self-Attn News Encoder (for NRMS)
class AttnNewsEncoder(nn.Module):
    def __init__(self, args, word_emb, cate_emb, ent_emb):
        super().__init__()
        self.args = args
        if 'ent_attn' not in self.args:
            self.args['ent_attn'] = False
        
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        self.self_attn = MultiHeadSelfAttention(word_emb.shape[1], 16, 16, 16)
        self.addi_attn = AttentionPooling(256, 200)
        self.dropout = nn.Dropout(0.2)
        
        if self.args['use_ent']:
            if args['ent_emb'] == 'transe':
                self.ent_embedding = nn.Embedding.from_pretrained(ent_emb)
            elif args['ent_emb'] == 'random':
                self.ent_embedding = nn.Embedding(ent_emb.shape[0], ent_emb.shape[1])
            
            if args['ent_attn']:
                self.ent_transformer = MultiHeadSelfAttention(ent_emb.shape[1], 16, 16, 16)
            elif args['ent_emb'] == 'avg':
                self.ent_fc1 = nn.Linear(word_emb.shape[1], 200)
                self.ent_fc2 = nn.Linear(200, 256)
            else:
                self.ent_fc1 = nn.Linear(ent_emb.shape[1], 200)
                self.ent_fc2 = nn.Linear(200, 256)
            self.ent_attn = AttentionPooling(256, 200)
            self.aggr_fc = nn.Linear(512, 256)
        else:
            self.aggr_fc = nn.Linear(256, 256)
    
    def forward(self, news, ents = None):
        title, body, cate = news[:, :30], news[:, 30: -2], news[:, -2:]
        ents, ent_words = ents[:, :, 0].squeeze(-1), ents[:, :, 1:] # (n_batch, n_ent), (n_batch, n_ent, n_ent_word)
        
        t_rep = self.word_embedding(title) # (n_batch, n_seq, emb_dim)
        t_rep = self.dropout(t_rep)
        t_rep = self.self_attn(t_rep, t_rep, t_rep) # (n_batch, n_seq, 256)
        t_rep = self.addi_attn(t_rep) # (n_batch, 256)
        
        if self.args['use_ent']:
            if self.args['ent_emb'] == 'avg':
                e_rep = self.word_embedding(ent_words) # (n_batch, n_ent, n_ent_word, emb_dim)
                e_rep = torch.mean(e_rep, dim = -2)
            else:
                e_rep = self.ent_embedding(ents) # (n_batch, n_ent, emb_dim)
                e_rep = self.dropout(e_rep)
            
            if self.args['ent_attn']:
                e_rep = self.ent_transformer(e_rep, e_rep, e_rep) # (n_batch, n_ent, 256)
            else:
                e_rep = F.relu(self.ent_fc1(e_rep))
                e_rep = self.ent_fc2(e_rep) # (n_news, n_ent, news_dim)
            e_rep = self.ent_attn(e_rep) # (n_batch, 256)
            t_rep = torch.cat((t_rep, e_rep), dim = -1)
        
        r = self.aggr_fc(t_rep)
        if 'aggr_relu' in self.args and self.args['aggr_relu']:
            r = F.relu(r)
        return r # (n_news, n_filter)

class AttnUserEncoder(nn.Module):
    def __init__(self, n_head, news_dim, query_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(news_dim, n_head, 16, 16)
        self.addi_attn = AttentionPooling(news_dim, query_dim)
    
    def forward(self, h): # (n_batch, n_news, 256)
        u = self.self_attn(h, h, h) # (n_batch, n_news, 256)
        u = self.addi_attn(u) # (n_batch, 256)
        return u

class Model(nn.Module):
    def __init__(self, args, word_emb, cate_emb, ent_emb):
        super().__init__()
        if args['model'] == 'NRMS':
            n_head, query_dim, news_dim = 16, 200, 256
            self.news_encoder = AttnNewsEncoder(args, word_emb, cate_emb, ent_emb)
            self.user_encoder = AttnUserEncoder(n_head, news_dim, query_dim)
    
    def forward(self, hist, samp, samp_ents = None, user_ents = None):
        n_batch, n_news, n_sequence = hist.shape
        n_samp = samp.shape[1] # k + 1
        n_ents = samp_ents.shape[2]
        n_ent_words = samp_ents.shape[3] # n_words + 1
        
        hist = hist.reshape(n_batch * n_news, n_sequence)
        if user_ents is not None:
            user_ents = user_ents.reshape(n_batch * n_news, n_ents, n_ent_words)
        h = self.news_encoder(hist, user_ents) # (n_batch*n_news, n_filter)
        h = h.reshape(n_batch, n_news, -1)  # (n_batch, n_news, n_filter)
        u = self.user_encoder(h) # (n_batch, n_filter)
        
        samp = samp.reshape(n_batch * n_samp, n_sequence)
        if samp_ents is not None:
            samp_ents = samp_ents.reshape(n_batch * n_samp, n_ents, n_ent_words)
        r = self.news_encoder(samp, samp_ents) # (n_batch*(k+1), n_filter)
        r = r.reshape(n_batch, n_samp, -1) # (n_batch, k + 1, n_filter)
        
        y = torch.bmm(r, u.unsqueeze(2)) # (n_batch, K + 1, 1)
        return y.squeeze(2)


