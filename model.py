import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, emb_dim, query_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, query_dim)
        self.fc2 = nn.Linear(query_dim, 1)
        
    def forward(self, x, mask = None):
        '''
        (n_batch, n_seq, emb_dim) -> (n_batch, emb_dim)
        a = q^T tanh(V * k + v)
        alpha = softmax(a)
        '''
        a = self.fc2(torch.tanh(self.fc1(x))) # (n_batch, n_seq, 1)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(a, dim = -2) # (n_batch, n_seq, 1)
        r = torch.matmul(alpha.transpose(-2, -1), x).squeeze(-2) # (n_batch, emb_dim)
        return r

class TextEncoder(nn.Module):
    def __init__(self, word_embedding, word_emb_dim, 
                 filter_num, window_size, query_dim, dropout, use_relu = False
                ):
        super().__init__()
        self.use_relu = use_relu
        self.word_embedding = word_embedding
        self.cnn = nn.Conv1d(word_emb_dim, filter_num, window_size, padding = window_size // 2)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.attn = AttentionPooling(filter_num, query_dim)
        
    def forward(self, x, mask = None):
        x_emb = self.word_embedding(x) # (n_batch, n_seq, emb_dim)
        x_emb = self.drop1(x_emb)
        x_rep = self.cnn(x_emb.transpose(2, 1)).transpose(2, 1) # (n_batch, n_seq, emb_dim)
        if self.use_relu:
            x_rep = F.relu(x_rep)
        x_rep = self.drop2(x_rep)
        x_rep = self.attn(x_rep, mask) # (n_batch, emb_dim)
        return x_rep

class CateEncoder(nn.Module):
    def __init__(self, cate_embedding, cate_emb_dim, out_dim, dropout = 0.2):
        super().__init__()
        self.cate_embedding = cate_embedding
        self.fc = nn.Linear(cate_emb_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x_emb = self.cate_embedding(x) # (n_batch, emb_dim)
        x_emb = self.drop(x_emb)
        x_rep = self.fc(x_emb) # (n_batch, out_dim)
        x_rep = F.relu(x_rep)
        return x_rep

class ConvNewsEncoder(nn.Module):
    def __init__(self, word_emb, cate_emb, 
                 filter_num, window_size, query_dim, dropout, args
                ):
        super().__init__()
        self.args = args
        if 'use_relu' not in args:
            args['use_relu'] = False
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        self.cate_embedding = nn.Embedding.from_pretrained(cate_emb)
        self.word_emb_dim = word_emb.shape[1]
        self.cate_emb_dim = cate_emb.shape[1]
        self.title_encoder = TextEncoder(self.word_embedding, self.word_emb_dim, 
                                 filter_num, window_size, query_dim, dropout, args['use_relu'])
        if args['use_body']:
            self.body_encoder = TextEncoder(self.word_embedding, self.word_emb_dim, 
                                     filter_num, window_size, query_dim, dropout, args['use_relu'])
            self.attn = AttentionPooling(filter_num, query_dim)
        if args['use_cate']:
            self.cate_encoder = CateEncoder(self.cate_embedding, self.cate_emb_dim, filter_num, dropout)
            self.subcate_encoder = CateEncoder(self.cate_embedding, self.cate_emb_dim, filter_num, dropout)
            self.attn = AttentionPooling(filter_num, query_dim)
    
    def forward(self, news):
        title, body, cate, subcate = news[:, :30], news[:, 30: -2], news[:, -2], news[:, -1] # max_title = 30
        
        r_t = self.title_encoder(title) # (n_news, emb_dim)
        
        if self.args['use_body'] and self.args['use_cate']:
            r_b = self.body_encoder(body) # (n_news, emb_dim)
            r_c = self.cate_encoder(cate) # (n_news, emb_dim)
            r_sc = self.subcate_encoder(subcate) # (n_news, emb_dim)
            r = torch.stack((r_t, r_b, r_c, r_sc), dim = 1) # (n_news, 4, emb_dim)
            r = self.attn(r) # (n_news, n_filter)
        elif self.args['use_body']:
            r_b = self.body_encoder(body) # (n_news, emb_dim)
            r = torch.stack((r_t, r_b), dim = 1) # (n_news, 4, emb_dim)
            r = self.attn(r) # (n_news, n_filter)
        elif self.args['use_cate']:
            r_c = self.cate_encoder(cate) # (n_news, emb_dim)
            r_sc = self.subcate_encoder(subcate) # (n_news, emb_dim)
            r = torch.stack((r_t, r_c, r_sc), dim = 1) # (n_news, 4, emb_dim)
            r = self.attn(r) # (n_news, n_filter)
        else:
            r = r_t
        return r # (n_news, n_filter)

class UserEncoder(nn.Module):
    def __init__(self, emb_dim, query_dim):
        super().__init__()
        self.attn = AttentionPooling(emb_dim, query_dim)
    
    def forward(self, h, mask = None): 
        u = self.attn(h, mask)
        return u

class NAML(nn.Module):
    def __init__(self, word_emb, cate_emb, args):
        super().__init__()
        filter_num, window_size, query_dim, dropout = 400, 3, 200, 0.2
        self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
        self.user_encoder = UserEncoder(filter_num, query_dim)
    
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

# NRMS
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True)  + 1e-8)
        
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.d_model = d_model # 300
        self.n_heads = n_heads # 20
        self.d_k = d_k # 20
        self.d_v = d_v # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads) # 300, 400
        
        self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, max_len, max_len) 
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        return context # (n_batch, n_seq, emb_dim)

class AttnNewsEncoder(nn.Module):
    def __init__(self, args, word_emb, cate_emb, n_head, news_dim, query_dim):
        super().__init__()
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        emb_dim = word_emb.shape[1]
        # self.self_attn = MultiHeadSelfAttention(n_head, emb_dim, query_dim, news_dim)
        self.self_attn = MultiHeadSelfAttention(emb_dim, n_head, 16, 16)
        self.addi_attn = AttentionPooling(news_dim, query_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, news):
        title, body, cate, subcate = news[:, :30], news[:, 30: -2], news[:, -2], news[:, -1]
        
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
    def __init__(self, word_emb, cate_emb, args):
        super().__init__()
        n_head, query_dim, news_dim = 16, 200, 256
        self.news_encoder = AttnNewsEncoder(word_emb, cate_emb, n_head, news_dim, query_dim)
        self.user_encoder = AttnUserEncoder(n_head, news_dim, query_dim)
    
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

# LSTUR
class GruUserEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 1, batch_first = True)

    def forward(self, h): # (n_batch, n_news, news_dim)
        h0 = torch.randn((1, h.shape[0], self.hidden_size), device = 'cuda')
        output, hn = self.gru(h, h0)
        return hn.squeeze(0) # (n_batch, news_dim)



# total
# class NewsEncoder(nn.Module):
#     def __init__(self, args, word_emb, cate_emb, ent_emb = None):
#         super().__init__()
#         self.args = args
#         self.word_embedding = nn.Embedding.from_pretrained(word_emb)
#         self.cate_embedding = nn.Embedding.from_pretrained(cate_emb)
#         word_emb_dim = word_emb.shape[1]
#         cate_emb_dim = cate_emb.shape[1]
#         n_head, query_dim = 16, 200
#         if args['model'] == 'NAML':
#             self.cnn = nn.Conv1d(word_emb_dim, filter_num, window_size, padding = window_size // 2)
#             self.attn = AttentionPooling(filter_num, query_dim)
#             if args['use_cate']:
#                 self.cate_fc1 = nn.Linear(cate_emb_dim, out_dim)
#                 self.cate_fc2 = nn.Linear(cate_emb_dim, out_dim)
#         if args['model'] == 'NRMS':
#             self.self_attn = MultiHeadSelfAttention(word_emb_dim, n_head, 16, 16)
#             self.addi_attn = AttentionPooling(256, 200)
#         if args['model'] == 'LSTUR':
#             filter_num, window_size, query_dim, dropout = 300, 3, 200, 0.2
#             args['use_relu'] = True
#             self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
#             self.user_encoder = GruUserEncoder(filter_num, filter_num)
#         if args['model'] == 'CAUM':
#             filter_num, window_size, query_dim, dropout = 400, 3, 200, 0.2
#             args['use_relu'] = False
#             self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
#             self.user_encoder = UserEncoder(filter_num, query_dim)
#         self.word_embedding = nn.Embedding.from_pretrained(word_emb)
#         emb_dim = word_emb.shape[1]
#         self.self_attn = MultiHeadSelfAttention(emb_dim, n_head, 16, 16)
#         self.addi_attn = AttentionPooling(news_dim, query_dim)
#         self.dropout = nn.Dropout(0.2)
    
#     def forward(self, news):
#         title, body, cate, subcate = news[:, :max_title], news[:, max_title: -2], news[:, -2], news[:, -1]
        
#         t_rep = self.word_embedding(title) # (n_batch, n_seq, emb_dim)
#         t_rep = self.dropout(t_rep)
#         if args['model'] == 'NAML':
#             x_rep = self.cnn(x_emb.transpose(2, 1)).transpose(2, 1) # (n_batch, n_seq, emb_dim)
#             if self.use_relu:
#                 x_rep = F.relu(x_rep)
#             x_rep = self.drop2(x_rep)
#             x_rep = self.attn(x_rep, mask) # (n_batch, emb_dim)
#             if args['use_cate']:
#                 x_emb = self.cate_embedding(x) # (n_batch, emb_dim)
#                 x_emb = self.drop(x_emb)
#                 x_rep = self.fc(x_emb) # (n_batch, out_dim)
#                 x_rep = F.relu(x_rep)
#         if args['model'] == 'NRMS':
#             t_rep = self.self_attn(t_rep, t_rep, t_rep) # (n_batch, n_seq, 256)
#             t_rep = self.addi_attn(t_rep) # (n_batch, 256)

class Model(nn.Module):
    def __init__(self, word_emb, cate_emb, args):
        super().__init__()
        if args['model'] == 'NAML':
            filter_num, window_size, query_dim, dropout = 400, 3, 200, 0.2
            args['use_relu'] = False
            self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
            self.user_encoder = UserEncoder(filter_num, query_dim)
        if args['model'] == 'NRMS':
            n_head, query_dim, news_dim = 16, 200, 256
            self.news_encoder = AttnNewsEncoder(word_emb, cate_emb, n_head, news_dim, query_dim)
            self.user_encoder = AttnUserEncoder(n_head, news_dim, query_dim)
        if args['model'] == 'LSTUR':
            filter_num, window_size, query_dim, dropout = 300, 3, 200, 0.2
            args['use_relu'] = True
            self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
            self.user_encoder = GruUserEncoder(filter_num, filter_num)
        if args['model'] == 'CAUM':
            filter_num, window_size, query_dim, dropout = 400, 3, 200, 0.2
            args['use_relu'] = False
            self.news_encoder = ConvNewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, args)
            self.user_encoder = UserEncoder(filter_num, query_dim)
    
    def forward(self, hist, samp, hist_ents = None, samp_ents = None):
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










