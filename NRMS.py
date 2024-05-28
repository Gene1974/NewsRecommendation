import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import torch
import torch.nn as nn
from AttentionModel import AttentionPooling, MultiHeadSelfAttention
assert(torch.cuda.is_available())

# Self-Attn News Encoder (for NRMS)
class AttnNewsEncoder(nn.Module):
    def __init__(self, word_emb, n_head, news_dim, query_dim):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        emb_dim = word_emb.shape[1]
        self.self_attn = MultiHeadSelfAttention(emb_dim, n_head, 16, 16)
        self.addi_attn = AttentionPooling(news_dim, query_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, news):
        title = news[:, :30]
        
        t_rep = self.word_embedding(title) # (n_batch, n_seq, emb_dim)
        t_rep = self.dropout(t_rep)
        t_rep = self.self_attn(t_rep, t_rep, t_rep) # (n_batch, n_seq, 256)
        t_rep = self.addi_attn(t_rep) # (n_batch, 256)
        
        return t_rep # (n_news, 256)

class AttnUserEncoder(nn.Module):
    def __init__(self, n_head, news_dim, query_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(news_dim, n_head, 16, 16)
        self.addi_attn = AttentionPooling(news_dim, query_dim)
    
    def forward(self, h): # (n_batch, n_news, 256)
        u = self.self_attn(h, h, h) # (n_batch, n_news, 256)
        u = self.addi_attn(u) # (n_batch, 256)
        return u

class NRMS(nn.Module):
    def __init__(self, word_emb):
        super().__init__()
        self.news_encoder = AttnNewsEncoder(word_emb, n_head = 16, news_dim = 256, query_dim = 200)
        self.user_encoder = AttnUserEncoder(n_head = 16, news_dim = 256, query_dim = 200)
    
    def forward(self, hist, samp):
        n_batch, n_news, n_sequence = hist.shape
        n_samp = samp.shape[1] # k + 1
        
        hist = hist.reshape(n_batch * n_news, n_sequence)
        h = self.news_encoder(hist) # (n_batch*n_news, n_filter)
        h = h.reshape(n_batch, n_news, -1)  # (n_batch, n_news, n_filter)
        u = self.user_encoder(h) # (n_batch, n_filter)
        
        samp = samp.reshape(n_batch * n_samp, n_sequence)
        r = self.news_encoder(samp) # (n_batch*(k+1), n_filter)
        r = r.reshape(n_batch, n_samp, -1) # (n_batch, k + 1, n_filter)
        
        y = torch.bmm(r, u.unsqueeze(2)) # (n_batch, K + 1, 1)
        return y.squeeze(2)