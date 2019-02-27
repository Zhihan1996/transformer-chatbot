import tensorflow as tf
import jieba
from Data_process import *
from Model import Transformer
from Hyperparams import hyperparams as hp
from tqdm import tqdm

vocab_fpath = './data/vocab.txt'

# sess = tf.Session()
# new_saver = tf.train.import_meta_graph('./model/iwslt2016_E25L1.98-88775.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./model'))

# def inference(query):
#     train_batch = get_batch_evaluate(query, vocab_fpath)
#     iter = tf.data.Iterator.from_structure(train_batch.output_types, train_batch.output_shapes)
#     xs, ys = iter.get_next()
#
#     train_init_op = iter.make_initializer(train_batch)
#
#     m = Transformer(hp)
#     pred = m.infer(xs, ys)
#
#     sess = tf.Session()
#     new_saver = tf.train.import_meta_graph('./model/iwslt2016_E25L1.98-88775.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
#     sess.run(train_init_op)
#     print(sess.run(pred))






def inference(query):
    # generate the id list for the input sentence
    word2idx, idx2word = load_vocab(vocab_fpath)

    query_id = []
    for word in jieba.cut(query):
        query_id.append(word2idx.get(word, 1))
    query_id.append(word2idx.get('<S>'))
    if len(query_id) >= hp.maxlen:
        query_id = query_id[:20]
    else:
        query_id = pad(query_id, hp.maxlen, vocab_fpath)
    query_input = [query_id]

    xs = (tf.constant(query_input), tf.constant(len(query_id)), tf.constant(query))

    # initialize a target sentence to run the model
    y = [[i for i in range(20)]]
    target = ''
    decoder_inputs = np.ones(shape=[1,1],dtype='int32')
    ys = (tf.constant(decoder_inputs), tf.constant(y), tf.constant(len(y[0])), tf.constant(target))

    # generate the input of model
    # x_types = (tf.int32, tf.int32, tf.string)
    # y_types = (tf.int32, tf.int32, tf.int32, tf.string)
    # x_shapes = ([None], (), ())
    # y_shapes = ([None], [None], (), ())

    # x_1 = tf.placeholder(tf.int32, [1,20], name='input')
    # x_2 = tf.placeholder(tf.int32, [1], name='input')
    # x_3 = tf.placeholder(tf.string, [1], name='input')
    #
    # y_1 = tf.placeholder(tf.int32, [1,1], name='output')
    # y_2 = tf.placeholder(tf.int32, [1,20], name='output')
    # y_3 = tf.placeholder(tf.int32, [1], name='output')
    # y_4 = tf.placeholder(tf.string, [1], name='output')
    #
    # x_input = (x_1, x_2, x_3)
    # y_input = (y_1, y_2, y_3, y_4)

    # feed_dict = {x_1:query_input, x_2:len(query_id), x_3:query, y_1:decoder_inputs, y_2:y, y_3:len(y[0]), y_4:target}

    with tf.Session() as sess:
        sess = tf.Session()
        # init = tf.global_variables_initializer()
        # sess.run(init)

        print(sess.run(xs))
        print(sess.run(ys))


        # m = Transformer(hp)
        # y_hat = m.infer(xs, ys)
        #
        # new_saver = tf.train.import_meta_graph('./model/iwslt2016_E25L1.98-88775.meta')
        # new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
        # rep = sess.run(y_hat)

    #print(sess.run(xs))
    #print(sess.run(ys))

    # print(sess.run(y_hat))
    # rep = sess.run(y_hat)
    reply = ''
    for id in rep[0]:
        if id == 1 or id == 4:
            break
        reply += idx2word.get(id)

    print(reply)

    # print(sess.run(y_hat, feed_dict=feed_dict))


