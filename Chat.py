import tensorflow as tf
import jieba
from Data_process import *
from Model import Transformer
from Hyperparams import hyperparams as hp
import signal
import sys

vocab_fpath = './data/vocab.txt'
word2idx, idx2word = load_vocab(vocab_fpath)

x_1 = tf.placeholder(tf.int32, [1,20], name='input')
x_2 = tf.placeholder(tf.int32, (), name='input')
x_3 = tf.placeholder(tf.string, (), name='input')

y_1 = tf.placeholder(tf.int32, [1,1], name='output')
y_2 = tf.placeholder(tf.int32, [1,19], name='output')
y_3 = tf.placeholder(tf.int32, (), name='output')
y_4 = tf.placeholder(tf.string, (), name='output')

x_input = (x_1, x_2, x_3)
y_input = (y_1, y_2, y_3, y_4)

sess = tf.Session()
m = Transformer(hp)
y_hat = m.infer(x_input, y_input)

new_saver = tf.train.Saver()
new_saver.restore(sess, tf.train.latest_checkpoint('./model'))


def generate_input(query):
    query_id = []
    for word in jieba.cut(query):
        query_id.append(word2idx.get(word, 1))
    query_id.append(word2idx.get('<S>'))
    if len(query_id) >= hp.maxlen:
        query_id = query_id[:20]
    else:
        query_id = pad(query_id, hp.maxlen, vocab_fpath)
    query_input = [query_id]

    y = [[i for i in range(hp.maxlen-1)]]
    target = ''
    decoder_inputs = np.ones(shape=[1,1],dtype='int32') * word2idx.get('<S>')

    return query_input, len(query_id), query, decoder_inputs, y, len(y[0]), target

def ids2sentence(ids):
    reply = ''
    for id in ids:
        if id == word2idx.get('</S>') or id == word2idx.get('<PAD>'):
            break
        reply += idx2word.get(id)
    return reply

if __name__ == "__main__":
    # load jieba package first
    a = jieba.cut('你好')
    print(a)
    while True:
        query = input('Enter Query: ')
        if len(query) == 0:
            sys.exit(0)

        query_input, x_length, query, decoder_inputs, y, y_length, target = generate_input(query)
        feed_dict = {x_1: query_input, x_2: x_length, x_3: query, y_1: decoder_inputs, y_2: y, y_3: y_length, y_4: target}
        ids = sess.run(y_hat, feed_dict=feed_dict)
        reply = ids2sentence(ids[0])
        print('>', ''.join(reply))


