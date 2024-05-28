import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionModel import AttentionPooling, ScaledDotProductAttention, MultiHeadSelfAttention
from NRMS import AttnNewsEncoder, AttnUserEncoder, NRMS
assert(torch.cuda.is_available())

# Use NRMS as BaseModel

class EntityNewsEncoder(AttnNewsEncoder):
    def __init__(self, args, word_emb, ent_emb):
        super().__init__(word_emb, 16, 256, 200)
        self.args = args
        self.ent_emb_mode = self.args.get('ent_emb', 'transe') # transe, trip, word_avg, random
        self.ent_attn_mode = self.args.get('ent_attn', 'attn')
        self.ent_trip_mode = self.args.get('ent_trip', 'attn') # attn, avg, fc
        print(''.format(self.ent_emb_mode, self.ent_emb_mode))

        if self.ent_emb_mode in ['transe', 'default']:
            self.ent_embedding = nn.Embedding.from_pretrained(ent_emb)
        elif self.ent_emb_mode == 'random':
            self.ent_embedding = nn.Embedding(ent_emb.shape[0], ent_emb.shape[1])
        elif self.ent_emb_mode == 'trip':
            self.ent_embedding = nn.Embedding.from_pretrained(ent_emb)
            if self.ent_trip_mode == 'attn':
                self.ent_trip_attn = MultiHeadSelfAttention(ent_emb.shape[1], 10, 10, 10)
            elif self.ent_trip_mode == 'fc':
                self.ent_trip_fc = nn.Sequential([
                    nn.Linear(ent_emb.shape[1], 200),
                    nn.ReLU(), 
                    nn.Linear(200, 100)
                ])
            self.ent_trip_pool = AttentionPooling(100, 200)
        self.ent_emb_size = ent_emb.shape[1]

        if self.ent_attn_mode:
            self.ent_attn = MultiHeadSelfAttention(ent_emb.shape[1], 16, 16, 16)
        else:
            self.ent_fc1 = nn.Linear(ent_emb.shape[1], 200)
            self.ent_fc2 = nn.Linear(200, 256)
        self.ent_aggr = AttentionPooling(256, 200)
        self.aggr_fc = nn.Linear(512, 256)
        
        if self.ent_emb_mode == 'word_avg':
            self.ent_aggr = nn.Sequential([
                nn.Linear(word_emb.shape[1], 200),
                nn.ReLU(), 
                nn.Linear(200, 256)
            ])
    
    def _gen_ent_rep(self, ents, ent_trips):
        # 加权方法：
        # 1. head + avg(tail)
        # 2. head + avg(relation + tail)
        # 3. head + attn(tail)
        head_reps = self.ent_embedding(ents) # (n_batch, emb_dim)
        tail_reps = []
        for tails in ent_trips: # [tail_id1, tail_id2, ...]
            tail_rep = self.ent_embedding(tails) # (n_trip, emb_dim)
            if self.ent_trip_mode == 'avg':
                tail_rep = torch.mean(tail_rep, dim = -2) # (emb_dim)
            elif self.ent_trip_mode == 'attn':
                tail_rep = self.ent_trip_attn(tail_rep, tail_rep, tail_rep) # (n_trip, emb_dim)
                tail_rep = self.ent_trip_pool(tail_rep) # (emb_dim)
            tail_reps.append(tail_rep)
        tail_reps = torch.stack(tail_reps) # (n_batch, emb_dim)
        return head_reps + tail_reps

    def forward(self, news, ents = None):
        '''
        mode:
            1. TransE(ent_emb)
            2. word embedding avg
            3. random
            4. graph embedding (TransE)
        '''
        title = news[:, :30]
        t_rep = self.word_embedding(title) # (n_batch, n_seq, emb_dim)
        t_rep = self.dropout(t_rep)
        t_rep = self.self_attn(t_rep, t_rep, t_rep) # (n_batch, n_seq, 256)
        t_rep = self.addi_attn(t_rep) # (n_batch, 256)
        
        ents, ent_words, ent_trips = ents[:, :, 0].squeeze(-1), ents[:, :, 1: 11], ents[:, :, 11:] # (n_batch, n_ent), (n_batch, n_ent, n_ent_word), (n_batch, n_ent, n_ent_wordtrip)
        
        if self.ent_emb_mode == 'word_avg': # word embedding average
            e_rep = self.word_embedding(ent_words) # (n_batch, n_ent, n_ent_word, emb_dim)
            e_rep = torch.mean(e_rep, dim = -2)
        elif self.ent_emb_mode == 'trip': # 三元组加权embedding
            e_rep = self._gen_ent_rep(ents, ent_trips)
        else:
            e_rep = self.ent_embedding(ents) # (n_batch, n_ent, emb_dim)
        e_rep = self.dropout(e_rep)
        
        if self.ent_attn_mode:
            e_rep = self.ent_attn(e_rep, e_rep, e_rep) # (n_batch, n_ent, 256)
        else:
            e_rep = F.relu(self.ent_fc1(e_rep))
            e_rep = self.ent_fc2(e_rep) # (n_news, n_ent, news_dim)
        e_rep = self.ent_aggr(e_rep) # (n_batch, 256)
        t_rep = torch.cat((t_rep, e_rep), dim = -1)
        
        r = self.aggr_fc(t_rep)

        # if 'aggr_relu' in self.args and self.args['aggr_relu']:
        #     r = F.relu(r)
        return r # (n_news, n_filter)

class EntityModel(NRMS):
    def __init__(self, args, word_emb, cate_emb, ent_emb):
        super().__init__(word_emb)
        self.news_encoder = EntityNewsEncoder(args, word_emb, ent_emb)

    def forward(self, impr_news, hist_news, impr_ents, hist_ents): # torch.Size([16, 5, 132]) torch.Size([16, 50, 132]) torch.Size([16, 5, 5, 20])
        # impr_news: (n_batch, n_samp, n_sequence) torch.Size([16, 5, 132])
        # hist_news: (n_batch, n_news, n_sequence) torch.Size([16, 50, 132]) 
        # impr_ents: (n_batch, n_samp, n_ent, n_ent_seq) torch.Size([16, 5, 5, 20])
        # hist_ents: (n_batch, n_news, n_ent, n_ent_seq) torch.Size([16, 50, 5, 20])
        n_batch, n_news, n_sequence = hist_news.shape
        n_batch, n_samp, n_ents, n_ent_seq = impr_ents.shape
        # n_samp = impr_news.shape[1] # k + 1
        
        hist_news = hist_news.reshape(n_batch * n_news, n_sequence)
        hist_ents = hist_ents.reshape(n_batch * n_news, n_ents, n_ent_seq)
        h = self.news_encoder(hist_news, hist_ents) # (n_batch*n_news, n_filter)
        h = h.reshape(n_batch, n_news, -1)  # (n_batch, n_news, n_filter)
        u = self.user_encoder(h) # (n_batch, n_filter)
        
        impr_news = impr_news.reshape(n_batch * n_samp, n_sequence)
        impr_ents = impr_ents.reshape(n_batch * n_samp, n_ents, n_ent_seq)
        r = self.news_encoder(impr_news, impr_ents) # (n_batch*(k+1), n_filter)
        r = r.reshape(n_batch, n_samp, -1) # (n_batch, k + 1, n_filter)
        
        y = torch.bmm(r, u.unsqueeze(2)) # (n_batch, K + 1, 1)
        return y.squeeze(2)



