import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentivePooling(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.actv1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x, mask = None):
        '''
        x: (n_batch, n_seq, emb_dim)
        mask: (n_batch, n_seq)
        a = q^T tanh(V * c + v)
        alpha = softmax(a)
        '''
        a = self.fc2(torch.tanh(self.fc1(x))) # (n_batch, n_seq, 1)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(a, dim = 1) # (n_batch, n_seq, 1)
        r = torch.bmm(alpha.transpose(1, 2), x).squeeze(1) # (n_batch, emb_dim)
        return r

class TextEncoder(nn.Module):
    def __init__(self, word_embedding, word_emb_dim, 
                 filter_num, window_size, query_dim, dropout
                ):
        super().__init__()
        self.word_embedding = word_embedding
        self.cnn = nn.Sequential(
            nn.Conv1d(word_emb_dim, filter_num, window_size, padding = window_size // 2),
            nn.ReLU()
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.attn = AttentivePooling(filter_num, query_dim)
        
    def forward(self, x, mask):
        x_emb = self.word_embedding(x) # (n_batch, n_seq, emb_dim)
        x_emb = self.drop1(x_emb)
        x_rep = self.cnn(x_emb.transpose(2, 1)).transpose(2, 1) # (n_batch, n_seq, emb_dim)
        x_rep = self.drop2(x_rep)
        x_rep = self.attn(x_rep, mask) # (n_batch, emb_dim)
        return x_rep

class CateEncoder(nn.Module):
    def __init__(self, cate_embedding, cate_emb_dim, out_dim, dropout):
        super().__init__()
        self.cate_embedding = cate_embedding
        self.fc = nn.Linear(cate_emb_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_emb = self.cate_embedding(x) # (n_batch, emb_dim)
        x_emb = self.drop(x_emb)
        x_rep = self.fc(x_emb) # (n_batch, out_dim)
        x_rep = F.relu(x_rep)
        return x_rep

class NewsEncoder(nn.Module):
    def __init__(self, word_emb, cate_emb, 
                filter_num, window_size, query_dim, dropout, n_title):
        super().__init__()
        self.n_title = n_title
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        self.cate_embedding = nn.Embedding.from_pretrained(cate_emb)
        self.word_emb_dim = word_emb.shape[1]
        self.cate_emb_dim = cate_emb.shape[1]
        self.title_encoder = TextEncoder(self.word_embedding, self.word_emb_dim, 
                                 filter_num, window_size, query_dim, dropout)
#         self.body_encoder = TextEncoder(self.word_embedding, self.word_emb_dim, 
#                                  filter_num, window_size, query_dim, dropout)
#         self.cate_encoder = CateEncoder(self.cate_emb_dim, filter_num)
#         self.subcate_encoder = CateEncoder(self.cate_emb_dim, filter_num)
#         self.attn = AttentivePooling(filter_num, query_dim)
    
    def forward(self, news, masks):
        title, body, cate, subcate = news[:, :self.n_title], news[:, self.n_title: -2], news[:, -2], news[:, -1]
        if masks is not None:
            title_mask, body_mask = masks[:, :self.n_title], news[:, self.n_title:]
        else:
            title_mask, body_mask = None, None
        
        r_t = self.title_encoder(title, title_mask) # (n_news, emb_dim)
#         r_b = self.body_encoder(body, body_mask) # (n_news, emb_dim)
#         r_c = self.body_encoder(cate) # (n_news, emb_dim)
#         r_sc = self.body_encoder(subcate) # (n_news, emb_dim)
        
#         r = torch.stack((r_t, r_b, r_c, r_sc), dim = 1) # (n_news, 4, emb_dim)
#         r = self.attn(r) # (n_news, n_filter)
        r = r_t
        return r # (n_news, n_filter)

class UserEncoder(nn.Module):
    def __init__(self, emb_dim, query_dim):
        super().__init__()
        self.attn = AttentivePooling(emb_dim, query_dim)
    
    def forward(self, h, mask): 
        u = self.attn(h, mask)
        return u

class NAML(nn.Module):
    def __init__(self, word_emb, cate_emb, filter_num, window_size, query_dim, dropout, n_title):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        self.cate_embedding = nn.Embedding.from_pretrained(cate_emb)
        self.word_emb_dim = word_emb.shape[1]
        self.cate_emb_dim = cate_emb.shape[1]
        self.news_encoder = NewsEncoder(word_emb, cate_emb, filter_num, window_size, query_dim, dropout, n_title)
        self.user_encoder = UserEncoder(filter_num, query_dim)
    
    def forward(self, hist, samp, h_mask = None, hist_news_mask = None, samp_news_mask = None): 
        '''
        h_mask: news level mask, (n_batch, n_news)
        hist_news_mask: word level mask, (n_batch, n_news, n_sequence)
        '''
        n_batch, n_news, n_sequence = hist.shape
        n_samp = samp.shape[1] # k + 1
        
        hist = hist.reshape(n_batch * n_news, n_sequence)
        if hist_news_mask is not None:
            hist_news_mask = hist_news_mask.reshape(n_batch * n_news, n_sequence - 2)
        h = self.news_encoder(hist, hist_news_mask) # (n_batch*n_news, n_filter)
        h = h.reshape(n_batch, n_news, -1)  # (n_batch, n_news, n_filter)
        u = self.user_encoder(h, h_mask) # (n_batch, n_filter)
        
        samp = samp.reshape(n_batch * n_samp, n_sequence)
        if samp_news_mask is not None:
            samp_news_mask = samp_news_mask.reshape(n_batch * n_samp, n_sequence - 2)
        r = self.news_encoder(samp, samp_news_mask) # (n_batch*(k+1), n_filter)
        r = r.reshape(n_batch, n_samp, -1) # (n_batch, k + 1, n_filter)
        
        y = torch.bmm(r, u.unsqueeze(2)) # (n_batch, K + 1, 1)
        return y.squeeze(2)






