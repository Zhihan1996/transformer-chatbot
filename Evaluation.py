import tensorflow as tf
import jieba
from Data_process import *
from Hyperparams import hyperparams as hp

vocab_fpath = '/home/zhihan/PycharmProjects/Research/transformer-chatbot/data/vocab.txt'

sess = tf.Session()

new_saver = tf.train.import_meta_graph('/home/zhihan/PycharmProjects/Research/transformer-chatbot/model/iwslt2016_E01L3.15-3551.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/home/zhihan/PycharmProjects/Research/transformer-chatbot/model'))
# saver = tf.train.Saver()
# saver.restore(sess, '/home/zhihan/PycharmProjects/Research/transformer-chatbot/model/checkpoint')

def inference(query):
    word2idx, idx2word = load_vocab(vocab_fpath)
    query_id = []
    for word in jieba.cut(query):
        query_id.append(word2idx.get(word, 1))
    query_id.append(word2idx.get('<S>'))
    if len(query_id) >= hp.maxlen:
        query_id = query_id[:20]
    else:
        query_id = pad(query_id, hp.maxlen, vocab_fpath)
    



inference('你叫什么名字')