import tensorflow as tf
import numpy as np
import os
import math
import logging
from Modules import *
from Data_process import *
from tqdm import tqdm
from Model import Transformer
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
from Params import params
import signal


path = './data'
vocab_fpath = './data/vocab.txt'

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
Params = params()
parser = Params.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(path, vocab_fpath, hp.batch_size, shuffle=True)

iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
#eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
#y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)


        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            # logging.info("# test evaluation")
            # _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            # summary_writer.add_summary(_eval_summaries, _gs)
            #
            # logging.info("# get hypotheses")
            # hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            #logging.info("# write results")
            model_output = "zzh2019_E%02dL%.2f" % (epoch, _loss)
            print ("epoch: %03d, loss: %.2f" % (epoch, _loss))
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            #translation = os.path.join(hp.evaldir, model_output)
            # with open(translation, 'w') as fout:
            #     fout.write("\n".join(hypotheses))

            #logging.info("# calc bleu score and append it to translation")
            #calc_bleu(hp.eval3, translation)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
