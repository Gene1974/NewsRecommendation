import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch

def logger(content):
    logging.getLogger('matplotlib.font_manager').disabled = True
    log_format = '[%(asctime)s] %(message)s'
    date_format = '%m%d %H:%M:%S'
    logging.basicConfig(level = logging.DEBUG, format = log_format, datefmt = date_format)
    logging.info(content)

def plot_result(avg_train_losses, avg_valid_losses):
    model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(avg_train_losses)
    plt.plot(avg_valid_losses)
    plt.legend(['train_loss', 'valid_loss'])
    plt.savefig('./train_loss/' + model_time + '.png')
    
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def count_number(data):
    length = [len(i) for i in data]
    print('max length: ', max(length))
    sorted_length = []
    occur = {}
    for i in length:
        if i in occur:
            occur[i] += 1
        else:
            occur[i] = 1
    s = sorted([[i, occur[i]] for i in occur], key = lambda x: x[0]) # 按长度排序（不是出现次数）
    return s

if __name__ == '__main__':
    count_number(['111', '1', '222'])
