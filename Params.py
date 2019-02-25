import argparse

class params:
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size', default=12487, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="/home/zhihan/PycharmProjects/Research/transformer-chatbot/model", help="log directory")
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--evaldir', default="/home/zhihan/PycharmProjects/Research/transformer-chatbot/model", help="evaluation dir")
    parser.add_argument('--vocab_fpath', default="/home/zhihan/PycharmProjects/Research/transformer-chatbot/data/vocab.txt", help="evaluation dir")

    parser.add_argument('--num_units', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=20, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")


