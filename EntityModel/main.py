import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import argparse
import time
import torch
from data import get_data
from EntityModel import EntityModel
from train import train_and_eval
assert(torch.cuda.is_available())

if __name__ == '__main__':
    print('begin time: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
    parser = argparse.ArgumentParser()
    # parser.add_argument('-model', default = 'NRMS')
    # training args
    parser.add_argument('-epochs', type = int, default = 6)
    # parser.add_argument('--no_ent', action="store_true")
    parser.add_argument('-times', type = int, default = 1)

    # entity model args
    parser.add_argument('-ent_emb', type = str, default = 'transe', help = 'ent emb mode')
    parser.add_argument('-ent_attn', type = str, default = 'attn', help = 'ent aggr mode')
    parser.add_argument('-ent_trip', type = str, default = 'attn', help = 'ent trips aggr mode')





    args = parser.parse_args().__dict__
    print('args:', args)
    for i in range(args['times']):
        train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, \
            word_emb, cate_emb, ent_emb = get_data()
        model = EntityModel(args, word_emb, cate_emb, ent_emb).to('cuda')
        train_and_eval(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, args['epochs'])
    
    
