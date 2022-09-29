import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

import argparse
import torch
from data import get_data
from EntityModel import Model
from pytorchtools import EarlyStopping
from train import train_and_eval
assert(torch.cuda.is_available())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('mode', default = 'train')
    parser.add_argument('-model', default = 'NRMS')
    # parser.add_argument('-time', default = '06151622', help = 'Test model')
    parser.add_argument('-epochs', type = int, default = 6)
    parser.add_argument('-use_ent', type = bool, default = True)
    parser.add_argument('-ent_emb', type = str, default = 'random')
    parser.add_argument('-times', type = int, default = 1)
    args = parser.parse_args().__dict__
    print(args)
    # args = {'model': 'NRMS', 'epochs': 6,
    #     'use_ent': True, 'ent_emb': 'avg'}
    for i in range(args['times']):
        train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, \
            word_emb, cate_emb, ent_emb = get_data()
        model = Model(args, word_emb, cate_emb, ent_emb).to('cuda')
        train_and_eval(model, train_dataset, dev_dataset, news_info, dev_users, dev_user_hist, news_ents, args['epochs'])
    
    
